'''
Deep Q Networks
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
    
    def act(self, state, env, epsilon):
        if random.random() > epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action  = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(env.action_space.n)
        return action

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, env, epsilon):
        if random.random() > epsilon:
            state   = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action



class DQN_Agent:
    def __init__(self, env, save_name, args):
        self.args = args
        self.save_name = save_name
        self.env = env
        self.env.seed(self.args.seed)
        if self.args.CNN:
            self.current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n).to(device)
            self.buffer_size = 100000
            self.replay_initial = 10000
        else:    
            self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
            self.buffer_size = 1000
            self.replay_initial = self.args.batch_size # NOTE: currently setting it to batch size. Can increase it to some higher value like 100
        self.optimizer = optim.Adam(self.current_model.parameters(), lr = self.args.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)  # buffer with original environment rewards
        # seeds
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        # tensorboardx
        if self.args.tensorboard:
                # print('Init tensorboardX')
                self.writer = SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    
    def compute_td_loss(self, batch):
        '''
        Compute the loss for the Q-networks
        '''

        state, action, reward, next_state, done = batch

        state      = Variable(torch.FloatTensor(np.float32(state))).to(device)
        next_state = Variable(torch.FloatTensor(np.float32(next_state))).to(device)
        action     = Variable(torch.LongTensor(action)).to(device)
        reward     = Variable(torch.FloatTensor(reward)).to(device)
        done       = Variable(torch.FloatTensor(done)).to(device)

        q_values      = self.current_model(state)
        next_q_values = self.current_model(next_state)

        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.args.gamma * next_q_value * (1 - done)
        
        loss = (q_value - (expected_q_value.data)).pow(2).mean()

        return loss

    def epsilon_scheduler(self,i,t,H):
        '''
        Return 0 if we want greedy actions
        Return 1 if we want to explore randomly for the remainder of the episode
        '''
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500
        
        if self.args.ct_func == 'linear':
            ct = t*H/self.args.T
        elif self.args.ct_func == 'neg_exp': # below the linear line
            ct = H*(1-np.exp(t*H/self.args.T))
        elif self.args.ct_func == 'exp': # above the linear line
            ct = H*(np.exp(t*np.log(2)/self.args.T)-1)
        elif self.args.ct_func == 'eps_greedy':
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * t / epsilon_decay)
            return epsilon
        if i < ct:
            # epsilon = 0
            return 0
        elif i>= ct:
            # epsilon = 1
            return 1
        # return epsilon
        

    def train(self):
        episode_reward = 0
        episode_num = 0
        episode_reward_buffer = []
        episode_len_buffer = []
        H = 0

        state = self.env.reset()
        episode_step = 0
        for frame_idx in range(self.args.T):
            episode_step += 1
            epsilon = self.epsilon_scheduler(episode_step,frame_idx,H)
            if self.args.tensorboard:
                self.writer.add_scalar('Epsilon',epsilon,frame_idx)
            action = self.current_model.act(state, self.env, epsilon)
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
                episode_len_buffer.append(episode_step)
                H = np.max(episode_len_buffer)  # increase the horizon length as it learns more
                episode_step = 0
            
            if len(self.replay_buffer) > self.replay_initial:
                batch = self.replay_buffer.sample(self.args.batch_size)
                loss = self.compute_td_loss(batch)
                # backward prop and update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log the loss
                if self.args.tensorboard:
                    self.writer.add_scalar('Loss',loss.item(),frame_idx)
            
            if len(episode_reward_buffer) > 100:
                # solved criteria for the environment
                if np.mean(np.array(episode_reward_buffer[-100:])) > self.args.env_max_rew or episode_num == self.args.max_episodes:
                # if episode_num == 1000:
                    np.save(self.save_name +'.npy',np.array(episode_reward_buffer))
                    break
