import os
import gym

from config import args

if __name__ == "__main__":
    if not os.path.exists('Exps/'):
        os.makedirs('Exps/')
    # name to save the .npy files for rewards

    if  args.algo == 'DQN':  
        from DQNVariants.dqn import DQN_Agent as Agent
    elif  args.algo == 'DDQN':  
        from DQNVariants.ddqn import DDQN_Agent as Agent
    elif  args.algo == 'DDDQN':  
        from DQNVariants.dddqn import DDDQN_Agent as Agent
    elif  args.algo == 'PrioritizedDQN':  
        from DQNVariants.prioritizedDQN import PrioritizedDQN_Agent as Agent

    if args.CNN:
        from DQNVariants.common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


    for k in range(args.num_exps):
        print('\t Experiment number: '+str(k))
        save_names = 'Exps/'+str(args.ct_func)+'_'+str(args.algo)+'_'+str(args.env_name[:-3])+'_'+str(args.H)+'_'+str(args.T)+'_'+str(k)
        if args.CNN:
            env    = make_atari(args.env_name)
            env    = wrap_deepmind(env)
            env    = wrap_pytorch(env)

        else: 
            env = gym.make(args.env_name)

        agent = Agent(env = env, save_name = save_names, args=args)
        agent.train()
