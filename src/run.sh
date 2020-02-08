# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export HOME="/storage/home/sidnayak"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory

source /tools/config.sh
# Activate your environment
source activate py35

# Change to the directory in which your code is present
cd /storage/home/sidnayak/Trajectory-length-based-Exploration/src/

# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
# python -u main.py --tensorboard=1 &> out
python3 -W ignore main.py --tensorboard=1 --num_exps=2 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 --H=200 --T=2000000 --ct_func=eps_greedy &> out
python3 -W ignore main.py --tensorboard=1 --num_exps=2 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 --H=200 --T=2000000 --ct_func=linear &> out
python3 -W ignore main.py --tensorboard=1 --num_exps=2 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 --H=200 --T=2000000 --ct_func=exp &> out
python3 -W ignore main.py --tensorboard=1 --num_exps=2 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 --H=200 --T=2000000 --ct_func=neg_exp &> out
