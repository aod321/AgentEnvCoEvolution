from random import random
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
from torch.utils.tensorboard import SummaryWriter


MAX_STEPS_PER_EPOSIDE = 2000

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
        env = WFCUnity3DEnv(wfc_size=9, file_name=gamepath, max_steps=MAX_STEPS_PER_EPOSIDE, random_seed=rank)
        return env
    return _init


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}.zip")
                  self.model.save(self.save_path)

        return True


if __name__ == "__main__":
    pcgworker = PCGWorker(9,9)
    # Create log dir
    current_time = datetime.now().strftime('%d-%m-%y-%H_%M')
    log_dir = f"./train_logs/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    # 参数定义
    gamepath = "/home/jiagpu7/0815_newbuild_linux_maxsteps/0815_newbuild_linux_maxsteps.x86_64"
    gamename = "0815_newbuild_linux_maxsteps.x86_64"
    num_env = 10  # Number of env to use
    # 是否在每轮进化最后评估一下所有历史环境环境
    EXTRA_EVAL = True
    TRAIN_EPOSIDES = 2000
    TRAIN_STEPS = 25000
    EVAL_EPOSIDES = 10
    REWARD_THREASHOLD = 0.5
    # Tensroboard log
    tb_logs_path = f"./runs/{current_time}"
    os.makedirs(tb_logs_path, exist_ok=True)
    writer = SummaryWriter(tb_logs_path)
    # 保险起见,测试的时候把以前的进程都杀掉
    print("killing all old processes")
    os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")
    print("Creating all train envs")
    env_list = []
    for i in range(num_env):
        env_list.append(make_env(i))
    vec_env = SubprocVecEnv(env_list)
    vec_env = VecMonitor(vec_env, log_dir)
    print("Creating eval env")
    eval_env = Monitor(WFCUnity3DEnv(wfc_size=9, file_name=gamepath, max_steps=2000))
    base_wave = vec_env.env_method(method_name="get_wave", indices=0)[0]
    eval_env.set_wave(base_wave)
    eval_env.render_in_unity()
    # Create the callback: check every 5000 steps
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    model = PPO('CnnPolicy', vec_env, verbose=0,  device=torch.device("cuda:2"))
    sum_evo_count = 0
    map_collections = []
    try:
        print("Evaluation before training: Random Agent")
        # random_agent vecenv评估, before training
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
        print(f'Done: Mean reward: {mean_reward} +/- {std_reward:.2f}')
        current_env = 0
        for i in tqdm(range(TRAIN_EPOSIDES)):
            print(f"Training: eposide: {i}/{TRAIN_EPOSIDES} for {TRAIN_STEPS} steps")
            # 1. 训练一定步数
            model.learn(total_timesteps=TRAIN_STEPS, callback=save_callback)
            print(f"Training Done for eposide: {i}/{TRAIN_EPOSIDES}, now evaluate for {EVAL_EPOSIDES} eposides")
            # 2. 评估一定轮数
            eval_env.set_wave(base_wave)
            eval_env.render_in_unity()
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPOSIDES, deterministic=False)
            print(f'Evaluatioon Done: Mean reward: {mean_reward} +/- {std_reward:.2f}')
            writer.add_scalar("eposide_mean_reward", mean_reward, global_step=i)
            # 3. 根据mean_reward选择进化方向
            # 3.1 平均reward >=0.5则从当前种子进化一张新地图,并且继续评估
            # 3.2 否则停止进化, 保留当前进化来的地图继续学习
            #-- 3.1
            evolve_count = 0
            while mean_reward > REWARD_THREASHOLD:
                print(f"Current map is too easy for agent now, genrating a new map, count for {evolve_count} times...")
                vec_env.env_method(method_name="mutate_a_new_map", base_wave=base_wave, indices=current_env)
                vec_env.env_method(method_name="render_in_unity", indices=current_env)
                temp_wave = vec_env.env_method(method_name="get_wave", indices=current_env)[0]
                eval_env.set_wave(temp_wave)
                eval_env.render_in_unity()
                print("Saving middle map to json file...")
                pcgworker.to_file(wave=temp_wave, filename=os.path.join(log_dir, f'{sum_evo_count}_{evolve_count}_{current_time}_middle.json'))
                print("Evaluating on this new map again ...")
                mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPOSIDES, deterministic=False)
                print(f'Evaluation Done: Mean reward on new map: {mean_reward} +/- {std_reward:.2f}')
                evolve_count +=1
            #-- 3.2
            if evolve_count > 0:
                print(f"{evolve_count} times evolution Done")
                sum_evo_count +=1
                print(f"All Evo times till now: {sum_evo_count}")
                print("Keep current map and continue training")
                # switch to another env window
                if current_env < num_env - 1:
                    current_env += 1
                else:
                    current_env = 0
                # set map to new env
                base_wave = temp_wave
                map_collections.append(temp_wave)
                vec_env.env_method(method_name="set_wave", wave=base_wave, indices=current_env)
                vec_env.env_method(method_name="render_in_unity", indices=current_env)
                print("Save current map to json file...")
                current_time =  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                pcgworker.to_file(wave=base_wave, filename=os.path.join(log_dir, f'{sum_evo_count}_{current_time}.json'))
                print("Save current model to file...")
                model.save(os.path.join(log_dir, f"{sum_evo_count}_{current_time}.pth"))
                print("Evaluating on current all map")
                mean_reward_1, std_reward_1 = evaluate_policy(model, vec_env, n_eval_episodes=EVAL_EPOSIDES * num_env, deterministic=False)
                print(f'Evaluation on all Done: Mean reward on all map: {mean_reward} +/- {std_reward:.2f}')
                if EXTRA_EVAL:
                    print("Extra evaluate on all old maps:")
                    extra_rewards_list = []
                    std_rewards_list = []
                    for i, iwave in enumerate(map_collections):
                        print(f"evaluating on map {i}:")
                        eval_env.set_wave(wave=iwave)
                        eval_env.render_in_unity()
                        temp_mean_rewards, temp_std = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=False)
                        std_rewards_list.append(temp_std)
                        extra_rewards_list.append(temp_mean_rewards)
                        writer.add_scalar(f"evo_rewards_{evolve_count}", temp_mean_rewards, global_step=i)
                    for i in range(extra_rewards_list):
                        print(f"mean reward on map{i}: {extra_rewards_list[i]}, std is : {std_rewards_list[i]}")
            else:
                print(f"Continue training without evolution")
    finally:
        current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        print("Save current model to file...")
        model.save(os.path.join(log_dir, "{current_time}_interuppted.pth"))
        vec_env.close()
        print("killing all old processes")
        os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")
                
