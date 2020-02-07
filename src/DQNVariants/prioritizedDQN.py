'''
Double Deep Q Networks
'''
import os
import math, random
import gym
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from collections import deque

# from ..config import args

# CUDA compatability
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class PrioritizedBuffer():
    def __init__(self, capacity, prob_alpha = 0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state,0)
        next_state = np.expand_dims(next_state,0)

        max_prior = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prior
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total    = len(self.buffer)
        weights  = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)
        
        batch       = zip(samples)
        states      = np.concatenate(batch[0])
        actions     = batch[1]
        rewards     = batch[2]
        next_states = np.concatenate(batch[3])
        dones       = batch[4]

        return states, actions, rewards, next_states, dones, indices, weights

    def __len__(self):
        return len(self.buffer)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prior in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prior



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon=0):
        # setting epsilon=0 because we do not want epsilon-greedy exploration
        if random.random() > epsilon:
            state   = (torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action  = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(env.action_space.n)
        return action



class PrioritizedDQN_Agent:
    def __init__(self, env_name, save_name, args):
        self.args = args
        self.save_name = save_name
        self.env = gym.make(args.env_name)
        self.env.seed(self.args.seed)
        self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.target_model =  DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.update_target()
        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = PrioritizedBuffer(100000)  # buffer with original environment rewards
        # seeds
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # tensorboardx
        if self.args.tensorboard:
                print('Init tensorboardX')
                self.writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    def update_target(self):
        self.target_model.load_state_dict(self.current_model.state_dict())
    
    def compute_td_loss(self, batch, grad = True):
        '''
        Compute the loss for the Q-networks
        '''

        state, action, reward, next_state, done, indices, weights = batch

        state      = Variable(torch.FloatTensor(np.float32(state)),requires_grad = grad)
        next_state = Variable(torch.FloatTensor(np.float32(next_state)),requires_grad= grad)
        action     = Variable(torch.LongTensor(action))
        reward     = Variable(torch.FloatTensor(reward))
        done       = Variable(torch.FloatTensor(done))
        weights    = Variable(torch.FloatTensor(weights))

        q_values      = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.args.gamma * next_q_value * (1 - done)
        
        priors = (q_value - (expected_q_value.detach())).abs()
        loss = priors* weights.mean()

        # evaluate gradient of loss wrt inputs for evaluating aux_rewards
        if grad:
            gradient = torch.autograd.grad(loss,state)
        else:
            gradient = None
        
        return loss, gradient, indices, priors

    def compute_td_loss_aux_rewards(self, batch,gradient,frame_idx):
        '''
        Store R = reward + aux_reward
        And compute the loss with aux_rewards
        '''
        aux_rew = torch.clamp(torch.norm(gradient[0],dim=1),min=0, max =10)
        new_reward = reward + self.args.eta * aux_rew.numpy()
        if self.args.tensorboard:
            self.writer.add_scalar('Rewards/aux_rew',aux_rew.mean(),frame_idx)
            self.writer.add_scalar('Rewards/rew',np.array(reward).mean(),frame_idx)
            self.writer.add_scalar('Rewards/rew',np.array(new_reward).mean(),frame_idx)

        loss, _, _, _ = self.compute_td_loss(batch,grad=False)
        return loss

        

    def train(self):
        frame_idx = 0
        episode_reward = 0
        episode_num = 0
        episode_reward_buffer = []

        # beta for prioritized buffer
        beta_start = 0.4
        beta_frames = 1000
        beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

        state = self.env.reset()

        while True:
            frame_idx += 1
            action = self.current_model.act(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                episode_num += 1
                state = self.env.reset()
                episode_reward_buffer.append(episode_reward)
                if self.args.tensorboard:
                    self.writer.add_scalar('Reward',episode_reward,episode_num)
                episode_reward = 0
            
            if len(self.replay_buffer) > self.args.batch_size:
                beta = beta_by_frame(frame_idx)
                batch = self.replay_buffer.sample(self.args.batch_size, beta)
                loss, gradient, indices, priors = self.compute_td_loss(batch, grad = self.args.grad_explore)
                # if gradient exploration method, then compute aux_rewards and the new loss with new rewards
                if self.args.grad_explore:
                    loss = self.compute_td_loss_aux_rewards(batch,gradient,frame_idx)

                # backward prop and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.replay_buffer.update_priorities(indices,priors.data.cpu().numpy())
                self.optimizer.step()

                # log the loss
                if self.args.tensorboard:
                    self.writer.add_scalar('Loss',loss.item(),frame_idx)
            
            if frame_idx % 100 == 0:
                self.update_target()
            
            if len(episode_reward_buffer) > 100:
                # solved criteria for the environment
                if np.mean(np.array(episode_reward_buffer[-100:])) > self.args.env_max_rew or episode_num == 1000:
                # if episode_num == 1000:
                    np.save(self.save_name +'.npy',np.array(episode_reward_buffer))
                    break


            
            

if __name__ == "__main__":
    if not os.path.exists('Exps/'):
        os.makedirs('Exps/')
    # name to save the .npy files for rewards
    if args.grad_explore:
        save_name = 'Exps/grad_expl_'
    else:
        save_name = 'Exps/vanilla_'

    for k in range(args.num_exps):
        save_name = save_name+'DQN_'+str(args.env_name[:-3])+'_'+str(args.eta)+'_'+str(k)
        agent = DQN_Agent(env_name = args.env_name, save_name = save_name, args=args)
        agent.train()
