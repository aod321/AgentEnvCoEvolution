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
from datetime import datetime
import numpy as np
from pcgworker.PCGWorker import *
from torch.utils.tensorboard import SummaryWriter
# from sb3_callbacks import SaveOnBestTrainingRewardCallback

if __name__ == "__main__":
    # restore = True
    # restore_path = "/home/yinzi/python_projects/AgentEnvCoEvolution/train_logs/16-08-22-11_44/{current_time}_interuppted.pth"
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

    pcgworker = PCGWorker(9,9)
    # Create log dir
    current_time = datetime.now().strftime('%d-%m-%y-%H_%M')
    # Tensroboard log
    tb_logs_path = f"./runs/{current_time}"
    os.makedirs(tb_logs_path, exist_ok=True)
    writer = SummaryWriter(tb_logs_path)
    # --
    log_dir = f"./train_logs/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    # 参数定义
    gamepath = "/home/yinzi/0815_newbuild_linux_maxsteps/0815_newbuild_linux_maxsteps.x86_64"
    gamename = "0815_newbuild_linux_maxsteps.x86_64"
    num_env = 5  # Number of env to use
    # 是否在每轮进化最后评估一下所有历史环境环境
    EXTRA_EVAL = True
    TRAIN_EPOSIDES = 2000
    TRAIN_STEPS = 25000
    EVAL_EPOSIDES = 30 * num_env
    REWARD_THREASHOLD = 0.6
    # 保险起见,测试的时候把以前的进程都杀掉
    print("killing all old processes")
    os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")
    print("Creating all train envs")
    env_list = []
    for i in range(num_env):
        env_list.append(make_env(i))
    if EXTRA_EVAL:
        eval_env = Monitor(WFCUnity3DEnv(wfc_size=9, file_name=gamepath, max_steps=2000))
    else:
        eval_env = None
    vec_env = SubprocVecEnv(env_list)
    vec_env = VecMonitor(vec_env, log_dir)
    # Create the callback: check every 5000 steps
    # save_callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir)
    model = PPO('CnnPolicy', vec_env, verbose=0,  device=torch.device("cuda:2"))
    map_collections = []
    sum_evo_count = 0
    try:
        print("Evaluation before training: Random Agent")
        # random_agent vecenv评估, before training
        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, deterministic=False)
        print(f'Done: Mean reward: {mean_reward} +/- {std_reward:.2f}')
        current_env = 0
        base_wave = vec_env.env_method(method_name="get_wave", indices=0)[0]
        map_collections.append(base_wave)
        for i in tqdm(range(TRAIN_EPOSIDES)):
            print(f"Training: eposide: {i}/{TRAIN_EPOSIDES} for {TRAIN_STEPS} steps")
            # 1. 训练一定步数
            # model.learn(total_timesteps=TRAIN_STEPS, callback=save_callback)
            model.learn(total_timesteps=TRAIN_STEPS)
            print(f"Training Done for eposide: {i}/{TRAIN_EPOSIDES}, now evaluate for {EVAL_EPOSIDES} eposides")
            # 2. 评估一定轮数
            mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=EVAL_EPOSIDES, deterministic=False)
            print(f'Evaluatioon Done: Mean reward: {mean_reward} +/- {std_reward:.2f}')
            writer.add_scalar("eposide_mean_reward", mean_reward, global_step=i)
            # 3. 根据mean_reward选择进化方向
            # 3.1 平均reward >=0.5则从当前种子进化一张新地图,并且继续评估
            # 3.2 否则停止进化, 保留当前进化来的地图继续学习
            #-- 3.1
            evolve_count = 0
            while mean_reward > REWARD_THREASHOLD:
            # while evolve_count < 2:
                print(f"Current map is too easy for agent now, genrating a new map, count for {evolve_count} times...")
                wave = vec_env.env_method(method_name="mutate_a_new_map", indices=current_env)[0]
                vec_env.env_method(method_name="render_in_unity", indices=current_env)
                print("Saving midlle map to json file...")
                pcgworker.to_file(wave=wave, filename=os.path.join(log_dir, f'{sum_evo_count}_{evolve_count}_{current_time}_middle.json'))
                print("Evaluating on all map again ...")
                mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=EVAL_EPOSIDES, deterministic=False)
                print(f'Evaluation Done: New Mean reward on all map: {mean_reward} +/- {std_reward:.2f}')
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
                base_wave = wave
                map_collections.append(wave)
                vec_env.env_method(method_name="set_wave", wave=wave, indices=current_env)
                vec_env.env_method(method_name="render_in_unity", indices=current_env)
                print("Save current map to json file...")
                current_time =  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
                pcgworker.to_file(wave=wave, filename=os.path.join(log_dir, f'{sum_evo_count}_{current_time}.json'))
                print("Save current model to file...")
                model.save(os.path.join(log_dir, f"{sum_evo_count}_{current_time}.pth"))
                if EXTRA_EVAL:
                    print("Extra evaluate on all old maps:")
                    extra_rewards_list = []
                    std_rewards_list = []
                    for i, wave in enumerate(map_collections):
                        print(f"evaluating on map {i}:")
                        eval_env.set_wave(wave=wave)
                        eval_env.render_in_unity()
                        temp_mean_rewards, temp_std = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=False)
                        std_rewards_list.append(temp_std)
                        extra_rewards_list.append(temp_mean_rewards)
                        writer.add_scalar(f"evo_rewards_{evolve_count}", temp_mean_rewards, global_step=i)
                    for i in range(len(extra_rewards_list)):
                        print(f"mean reward on map{i}: {extra_rewards_list[i]}, std is : {std_rewards_list[i]}")
            else:
                print(f"Continue training without evolution")
    finally:
        current_time =  time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
        # print("Save current model to file...")
        model.save(os.path.join(log_dir, f"{current_time}_interuppted.pth"))
        vec_env.close()
        print("killing all old processes")
        os.system(f"nohup pidof {gamename} | xargs kill -9> /dev/null 2>&1 & ")
                
