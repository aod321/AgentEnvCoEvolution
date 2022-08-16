import enum
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from WFCUnity3DEnv import WFCUnity3DEnv
from typing import Callable
from tqdm import tqdm
import os
import gym
import torch
from stable_baselines3 import A2C, PPO
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from datetime import datetime
import numpy as np
from pcgworker.PCGWorker import *
from glob import glob
# from natsort import sorted
import re



if __name__ == "__main__":
    try:
        # 保险起见,测试的时候把以前的进程都杀掉
        print("killing all old processes")
        os.system(f"nohup pidof evalbuild_linux_maxsteps | xargs kill -9> /dev/null 2>&1 & ")
        gamepath = "/home/yinzi/eval_env_build/evalbuild_linux_maxsteps"
        log_path = "./train_logs"
        times = "16-08-22-16_44"
        env = Monitor(WFCUnity3DEnv(wfc_size=9, file_name=gamepath, max_steps=2000))
        model_path = os.path.join(log_path, times, "3_2022-08-16-17:24:07.pth")
        model = PPO.load(model_path, env=env)
        wfcworker = PCGWorker(9, 9)
        json_path  = sorted(glob(os.path.join(log_path,times,r"*.json")))
        map_path = []
        for jpath in json_path:
            if len(re.findall(".*midlle.*", jpath)) == 0:
                map_path.append(jpath)
        reawrd_list = []
        std_list = []
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False)
        print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
        reawrd_list.append(mean_reward)
        std_list.append(std_reward)
        eposide_rewards_list = []
        for i, map in enumerate(map_path):
            print(f"evaluating on map {i+1}:")
            wave = wfcworker.from_file(map)
            env.set_wave(wave=wave)
            env.render_in_unity()
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False)
            # eposide_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=False, return_episode_rewards=True)
            # eposide_rewards_list.append(eposide_reward)
            reawrd_list.append(mean_reward)
            std_list.append(std_reward)
        for i in range(len(eposide_rewards_list)):
            # eposide_reward = eposide_rewards_list[i]
            mean_reward = reawrd_list[i]
            std_reward = std_list[i]
            print(f'map {i} Mean reward: {mean_reward} +/- {std_reward:.2f}')
            # print(f'map {i}: mean: {np.mean(eposide_reward[0])}, \n all: {eposide_reward}')

    finally:
        print("killing all old processes")
        os.system(f"nohup pidof evalbuild_linux_maxsteps | xargs kill -9> /dev/null 2>&1 & ")
        
    



