import sys
sys.path.append('../')
from dm_env_rpc.v1 import dm_env_adaptor
import _load_environment as dm_tasks
from gym_wrapper_jump import GymFromDMEnv
import numpy as np
import einops
from pcgworker.PCGWorker import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO
from dm_env_rpc.v1 import tensor_utils
from dm_env_rpc.v1 import dm_env_rpc_pb2
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
import argparse
from stable_baselines3.common.env_checker import check_env
import torch
import uuid as uuid_lib
import os
from PIL import Image

# os.system("export GRPC_ENABLE_FORK_SUPPORT=1")
uuid = str(uuid_lib.uuid4())[:5]
print(f"current UUID:{uuid}")
parser = argparse.ArgumentParser(description='Training Pipeline')
parser.add_argument('--train_eposides', dest='train_eposides', default=2000, type=int)
parser.add_argument('--train_steps', dest='train_steps', default=2000, type=int)
parser.add_argument('--evlaute_eposide', dest='evlaute_eposide', default=10, type=int)
parser.add_argument('--evlaute_steps', dest='evlaute_steps', default=2000, type=int)
parser.add_argument('--evol_evaluate_steps', dest='evol_evaluate_steps', default=2000, type=int)
parser.add_argument('--test', dest='test', default=False, type=bool)
parser.add_argument('--start_evolution', dest='start_evolution', default="auto", type=str)
parser.add_argument('--keep_evolution', dest='keep_evolution', default="auto", type=str)
parser.add_argument('--restore', dest='restore', default=False, type=bool)

args = parser.parse_args()
print(args)
train_eposides = args.train_eposides
train_steps = args.train_steps
evlaute_steps = args.evlaute_steps
evol_evaluate_steps = args.evol_evaluate_steps
restore = args.restore
evlaute_eposide = args.evlaute_eposide


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
        self.save_path = os.path.join(log_dir)
        os.makedirs(self.save_path, exist_ok=True)
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

            #   self.model.save(os.path.join(self.save_path, f"{self.num_timesteps}_{mean_reward:.2f}.zip"))
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print(f"Saving new best model to {self.save_path}/best_model_{uuid}.zip")
                  self.model.save(os.path.join(self.save_path, f"best_model_{uuid}.zip"))
        return True



class Pipeline():
    def __init__(self) -> None:
        self.LOGDIR = "./new_train_logs"
        self.IMG_DIR = "./new_train_logs/images"
        self.JSON_DIR = "./new_train_logs/jsons"
        self.CHECK_FREQ = 1000
        os.makedirs(self.LOGDIR, exist_ok=True)
        os.makedirs(self.IMG_DIR, exist_ok=True)
        os.makedirs(self.JSON_DIR, exist_ok=True)
        self.TASK_OBSERVATIONS = ['RGBA_INTERLEAVED', 'reward', 'done']
        self.PORT = 30051
        # create worker
        self.PCGWorker_ = PCGWorker(9,9)
        # start from empty aera
        self.wave = self.PCGWorker_.build_wave()
        # inital connectiy from probility space
        self._SPACE = self.get_space_from_wave(self.wave)
        self._SEED = np.ones((81,1,2)).astype(np.int32)
        self.Unity_connection_details = None
        self.world_name = None
        self.dm_env = None
        self.gym_env = None
        # initial empty space world
        self.create_and_join_world()
        self.best_mean = -np.inf
        self.model = PPO("CnnPolicy", self.gym_env, verbose=1)
        if restore:
            try:
                print("Loading Parmaters")
                self.model.load(os.path.join(self.LOGDIR, f"last_model_{uuid}.zip"))
            except Exception as e:
                print(e)
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=self.CHECK_FREQ, log_dir=self.LOGDIR)
        self.step_rewards = []
    
    def get_space_from_wave(self, wave=None):
        if not wave:
            wave = self.wave
        mask , _ = self.PCGWorker_.connectivity_analysis(wave = wave, visualize_ = False, to_file = False)
        # reduce mask to 9x9 for processing
        reduced_map = einops.reduce(mask,"(h a) (w b) -> h w", a=20, b=20, reduction='max').reshape(-1)
        # use maxium playable area as probility space
        return np.flatnonzero(reduced_map == np.argmax(np.bincount(reduced_map))).astype(np.int32)
    
    def save_img_json_from_wave(self, id, wave=None):
        if not wave:
            wave = self.wave
        canvas =  self.PCGWorker_.render(wave, wind_name = "canvas",write_ = False,write_id = 0,output = True, verbose = False)
        Image.fromarray(canvas).save(os.path.join(self.IMG_DIR, f"{id}_{uuid}.png"))
        self.PCGWorker_.to_file(wave=wave, filename=os.path.join(self.JSON_DIR, f"{id}_{uuid}.json"))

    def create_and_join_world(self):
        try:
            self.Unity_connection_details, self.world_name = dm_tasks._connect_to_environment(self.PORT, 
                                    create_world_settings={"seed": self._SEED},
                                    join_world_settings={
                                                        "agent_pos_space": self._SPACE,
                                                        "object_pos_space": self._SPACE
                                                        }
                                    )
            self.dm_env = dm_tasks._DemoTasksProcessEnv(self.Unity_connection_details, self.TASK_OBSERVATIONS, num_action_repeats=1)
            self.gym_env = GymFromDMEnv(self.dm_env)
            print(GymFromDMEnv(self.dm_env))
            self.gym_env = Monitor(GymFromDMEnv(self.dm_env), self.LOGDIR)
            check_env(self.gym_env)

            
        except Exception as e:
                print("Reset Unity Map World Failed")
                raise e
    
    def evaluate(self, maxeposides,maxsteps, evolution="auto"):
        print(f"Evaluation for {maxeposides} episodes")
        self.step_rewards = []
        for _ in tqdm(range(maxeposides)):
            if evolution == "never":
                print("Reward always 0")
                # 保留测试用: reward都为0, 就会一直继续训练
                reward = 0
            elif evolution == "always":
                print("Reward always 1")
                    # 保留测试用: reward都为1, 就会一直跳到进化过程
                reward = 1
            else:
                # 真实评估
                done = False
                obs = self.gym_env.reset()
                for _ in range(maxsteps):
                    action, _states = self.model.predict(obs, deterministic=False)
                    obs, reward, done, info = self.gym_env.step(action)
                    if done:
                        break
            self.step_rewards.append(reward)
        print(f"step_rewards:{self.step_rewards}")
        return self.step_rewards

    def run(self, test=False):
        try:
            for eposide in tqdm(range(train_eposides)):
                self.save_img_json_from_wave(self.wave)
                print(f"Training: eposide: {eposide}/{train_eposides} for {train_steps} steps")
                if not test:
                    self.model.learn(total_timesteps=train_steps, callback=self.callback)
                #else: 测试的时候不真实训练模型
                print(f"Evaluating for {evlaute_steps} steps")
                evolution = "auto" if not test else args.start_evolution
                self.evaluate(maxeposides=evlaute_eposide, maxsteps=evlaute_steps, evolution=evolution)
                mean_reward = np.mean(self.step_rewards)
                print(f"Evaluation: train_eposide: {eposide}/{train_eposides}, mean_reward: {mean_reward}")
                # Using wfc to mutate a new env unitl not more than half of the eposides are rewarded
                evolve_count = 0
                if(np.mean(self.step_rewards) < 0.5):
                    print("Continue training on old map...")
                    continue
                mutated_wave = self.wave
                while np.mean(self.step_rewards) > 0.5:
                    evolve_count += 1
                    print(f"Need to evolve, already evolved for: {evolve_count} times")
                    print("Genrating a new map...")
                    mutated_wave = self.PCGWorker_.mutate(mutated_wave, 81)
                    self._SPACE = self.get_space_from_wave(mutated_wave)
                    self._SEED = np.array(mutated_wave.wave_oriented).astype(np.int32)
                    # Recreate and join Unity Map World
                    self.create_and_join_world()
                    print("Evlauating on new map...")
                    evolution = "auto" if not test else args.keep_evolution
                    self.evaluate(maxeposides=evlaute_eposide, maxsteps=evol_evaluate_steps, evolution=evolution)
                    print("Evlauate Done")
                    print(f"mean eposide_rewards on new map: \n {np.mean(self.step_rewards)}")
                # keep new map seed
                self.wave = mutated_wave
                self._SPACE = self.get_space_from_wave(mutated_wave)
                self._SEED = np.array(mutated_wave.wave_oriented).astype(np.int32)
                # Unity render new training map
                print("Unity render new training map...")
                self.create_and_join_world()
                print("Training on new map...")
        finally:
            print("saving checkpoint...")
            self.model.save(os.path.join(self.LOGDIR, f"last_model_{uuid}.zip"))
    

if __name__ == '__main__':
    trainer = Pipeline()
    trainer.run(test=args.test)


