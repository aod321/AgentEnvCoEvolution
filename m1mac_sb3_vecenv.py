import sys
sys.path.append("../")
import time
import gym
import numpy as np
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from pcgworker.PCGWorker import *
import matplotlib.pyplot as plt
import AgentEnvCoEvolution._load_environment as dm_tasks
import torch
from typing import Callable
import uuid as uuid_lib
import argparse
from AgentEnvCoEvolution.WFCUnity3DEnv import WFCUnity3DEnv
import sys
import argparse
import uuid as uuid_lib
import os
from PIL import Image
import copy
import time


#  export PYTORCH_ENABLE_MPS_FALLBACK=1
def make_env(rank: int) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = WFCUnity3DEnv(wfc_size=9, file_name=args.gamepath)
        return env
    return _init


if __name__ == '__main__':
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # os.system("export GRPC_ENABLE_FORK_SUPPORT=1")
    uuid = str(uuid_lib.uuid4())[:5]
    print(f"current UUID:{uuid}")
    parser = argparse.ArgumentParser(description='Training Pipeline')
    parser.add_argument('--train_eposides', dest='train_eposides', default=2000, type=int)
    parser.add_argument('--train_steps', dest='train_steps', default=2000, type=int)
    parser.add_argument('--evlaute_eposide', dest='evlaute_eposide', default=10, type=int)
    parser.add_argument('--evlaute_steps', dest='evlaute_steps', default=2000, type=int)
    parser.add_argument('--evol_evaluate_steps', dest='evol_evaluate_steps', default=2000, type=int)
    parser.add_argument('--gamepath', dest='gamepath', default="/Users/yinzi/Downloads/0815_mac_build/0815_mac_build.app/Contents/MacOS/tilemap_render", type=str)
    args = parser.parse_args()

    vec_env = None
    try:
        print("killing all old processes")
        os.system("nohup pidof tilemap_render | xargs kill -9> /dev/null 2>&1 & ")
        num_env = 4  # Number of env to use
        current_env = 0
        # By default, we use a DummyVecEnv as it is usually faster (cf doc)
        print("loading all envs")
        env_list = []
        for i in range(num_env):
            env_list.append(make_env(i))
        vec_env = SubprocVecEnv(env_list)
        # model = A2C('CnnPolicy', vec_env, verbose=0,  device=DEVICE)
        model = DQN('CnnPolicy', vec_env, verbose=0,  device=DEVICE)
        print("trainning")
        n_timesteps = 2000
        # # Multiprocessed RL Training
        start_time = time.time()
        model.learn(n_timesteps)
        total_time_multi = time.time() - start_time

        print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")
        # # Select a environment for evaluation
        # eval_env = env_list[current_env]
        # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        # print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    finally:
        if vec_env:
            vec_env.close()
        print("killing all old processes")
        os.system("nohup pidof tilemap_render | xargs kill -9> /dev/null 2>&1 & ")
