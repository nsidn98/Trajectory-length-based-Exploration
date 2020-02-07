'''
Dueling Double Deep Q Networks
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


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)


class DDDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DDDQN, self).__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage - advantage.mean()
    
    def act(self, state, epsilon=0):
        # setting epsilon=0 because we do not want epsilon-greedy exploration
        if random.random() > epsilon:
            state   = (torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action  = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(env.action_space.n)
        return action

class CnnDDDQN(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super(CnnDDDQN, self).__init__()
        
        
        self.input_shape = input_shape
        self.num_actions = num_outputs
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon=0):
        if random.random() > epsilon:
            state   = (torch.FloatTensor(np.float32(state)).unsqueeze(0)).to(device)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action


class DDDQN_Agent:
    def __init__(self, env, save_name, args):
        self.args = args
        self.save_name = save_name
        self.env = env
        self.env.seed(self.args.seed)
        if self.args.CNN:
            self.current_model = CnnDDDQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
            self.target_model = CnnDDDQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
            self.buffer_size = 100000
            self.replay_initial = 10000
        else:    
            self.current_model = DDDQN(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
            self.target_model = DDDQN(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
            self.buffer_size = 1000
            self.replay_initial = self.args.batch_size # NOTE: currently setting it to batch size. Can increase it to some higher value like 100
        self.update_target()
        self.optimizer = optim.Adam(self.current_model.parameters(), lr = self.args.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)  # buffer with original environment rewards
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

        state, action, reward, next_state, done = batch

        state      = Variable(torch.FloatTensor(np.float32(state)),requires_grad = grad).to(device)
        next_state = Variable(torch.FloatTensor(np.float32(next_state)),requires_grad= grad).to(device)
        action     = Variable(torch.LongTensor(action)).to(device)
        reward     = Variable(torch.FloatTensor(reward)).to(device)
        done       = Variable(torch.FloatTensor(done)).to(device)

        q_values      = self.current_model(state)
        next_q_values = self.target_model(next_state)
        

        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.args.gamma * next_q_value * (1 - done)
        
        loss = (q_value - (expected_q_value.detach())).pow(2).mean()

        # evaluate gradient of loss wrt inputs for evaluating aux_rewards
        if grad:
            gradient = torch.autograd.grad(loss,state)
        else:
            gradient = None
        
        return loss, gradient

    def compute_td_loss_aux_rewards(self, batch,gradient,frame_idx):
        '''
        Store R = reward + aux_reward
        And compute the loss with aux_rewards
        '''
        state, action, reward, next_state, done = batch
        if self.args.CNN:
            aux_rew = torch.clamp(torch.norm(gradient[0],dim=1),min=0, max =10)
            aux_rew = torch.clamp(torch.norm(aux_rew,dim=1),min=0, max =10)
            aux_rew = torch.clamp(torch.norm(aux_rew,dim=1),min=0, max =10)
        else:
            aux_rew = torch.clamp(torch.norm(gradient[0],dim=1),min=0, max =10)
        # print(type(aux_rew.cpu().numpy().mean()))
        # print(aux_rew.cpu().numpy())
        # print(aux_rew.cpu().numpy().mean())
        new_reward = reward + self.args.eta * aux_rew.cpu().numpy()
        if self.args.tensorboard:
            # self.writer.add_scalar('Rewards/aux_rew',aux_rew.cpu().numpy().mean(),frame_idx)
            self.writer.add_scalar('Rewards/aux_rew',np.array(new_reward).mean()-np.array(reward).mean(), frame_idx)
            self.writer.add_scalar('Rewards/rew',np.array(reward).mean(),frame_idx)
            self.writer.add_scalar('Rewards/new_rew',np.array(new_reward).mean(),frame_idx)
        
        batch = state, action, new_reward, next_state, done

        loss, _ = self.compute_td_loss(batch,grad=False)
        return loss

        

    def train(self):
        frame_idx = 0
        episode_reward = 0
        episode_num = 0
        episode_reward_buffer = []

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
            
            if len(self.replay_buffer) > self.replay_initial:
                batch = self.replay_buffer.sample(self.args.batch_size)
                loss, gradient = self.compute_td_loss(batch, grad = self.args.grad_explore)
                # if gradient exploration method, then compute aux_rewards and the new loss with new rewards
                if self.args.grad_explore:
                    loss = self.compute_td_loss_aux_rewards(batch,gradient,frame_idx)

                # backward prop and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log the loss
                if self.args.tensorboard:
                    self.writer.add_scalar('Loss',loss.item(),frame_idx)
            
            if frame_idx % 100 == 0:
                self.update_target()
            
            if len(episode_reward_buffer) > 100:
                # solved criteria for the environment
                if np.mean(np.array(episode_reward_buffer[-100:])) > self.args.env_max_rew or episode_num == self.args.max_episodes:
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
        agent = DDDQN_Agent(env_name = args.env_name, save_name = save_name, args=args)
        agent.train()
