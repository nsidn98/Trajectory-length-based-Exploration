import argparse

def str2list(v):
    if not v:
        return v
    else:
        return [v_ for v_ in v.split(',')]

'''
python3 main.py --tensorboard=1 --grad_explore=0 --env_name=CartPole-v0 --env_max_rew=195 --num_exps=1 --algo=DDDQN
python3 main.py --tensorboard=1 --grad_explore=1 --env_name=CartPole-v0 --env_max_rew=195 --num_exps=1 --algo=DDDQN

python3 main.py --tensorboard=1 --grad_explore=0 --env_name=MountainCar-v0 --env_max_rew=-110 --num_exps=1 --algo=DDDQN
python3 main.py --tensorboard=1 --grad_explore=1 --env_name=MountainCar-v0 --env_max_rew=-110 --num_exps=1 --algo=DDDQN

python3 main.py --tensorboard=1 --grad_explore=0 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --num_exps=1 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=10000
python3 main.py --tensorboard=1 --grad_explore=1 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --num_exps=1 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=10000

'''

parser = argparse.ArgumentParser(description='DQN')

parser.add_argument('--seed', type=int, default=0,
                    help='random seed for gym, torch, cuda, numpy')
parser.add_argument('--tensorboard', type=int, default=1, 
                    help='Whether we want tensorboardX logging')
parser.add_argument('--grad_explore', type=int, default=1,
                    help='Whether we want to use the gradient exploration method')
parser.add_argument('--eta', type = float, default=0.1,
                    help = 'scaling factor for auxillary reward')
parser.add_argument('--num_exps', type=int, default=1,
                    help='Number of experiments to run')
parser.add_argument('--env_name',type=str, default='CartPole-v0',
                    help='Environment to run experiments on')
parser.add_argument('--env_max_rew', type=int, default = 195,
                    help='max reward avg to be achieved to be considered as solved') # cartpole=195, mountaincar=-110, PongNoFrameskip-v4=18
parser.add_argument('--gamma', type=float, default=0.99,
                    help=' Discount for rewards')
parser.add_argument('--batch_size', type=int, default=32,
                    help=' Batch Size')
parser.add_argument('--algo', type=str, default='DDDQN',
                    help='algorithm to use')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate for ADAM')
parser.add_argument('--CNN', type=int, default=0,
                    help='To use CNN for Atari environments')
parser.add_argument('--max_episodes', type=int, default=1000,
                    help='Maximum number of episodes to train the agent')


args = parser.parse_args()
