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
cd /storage/home/sidnayak/Gradient-Exploration-for-Deep-RL/src/

# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
# python -u main.py --tensorboard=1 &> out
# python -u -W ignore main.py --tensorboard=1 --grad_explore=0 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --num_exps=5 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 &> out
python -u -W ignore main.py --tensorboard=1 --grad_explore=1 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --num_exps=5 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 --eta=0.01 &> out
python -u -W ignore main.py --tensorboard=1 --grad_explore=1 --env_name=PongNoFrameskip-v4 --env_max_rew=18 --num_exps=5 --algo=DDDQN --CNN=1 --lr=0.0001 --max_episodes=300 --eta=0.1 &> out
