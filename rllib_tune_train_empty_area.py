import os
import ray
import sys
sys.path.append('../')
import einops
import argparse
import numpy as np
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.impala as impala
from _load_environment import EnvironmentSettings, load_from_disk
import ray.rllib.agents.dqn as dqn
from gym_wrapper import GymFromDMEnv
from pcgworker.PCGWorker import *


# argparser
parser = argparse.ArgumentParser(description='RLlib multi-thread sampling Unity3D envs example')
parser.add_argument('--algrithm', default="IMPALA", type=str, help='algrithm: IMPALA, PPO, DQN')
parser.add_argument('--game', default="/Users/yinzi/Downloads/test/m1_map_rpc_built_x64.app/Contents/MacOS/tilemap_render", type=str, help='path to built unity runtime')
parser.add_argument('--workers', default=4, type=int, help='workers number: 1, 4, 8, 16')
parser.add_argument('--train_batch_size', default=2048, type=int, help='batch_size for training')
parser.add_argument('--rollout_fragment_length', default=512, type=int, help='batch_size for training')
parser.add_argument('--timescale', default=2, type=int, help='timescale for unity3d timesys')
parser.add_argument('--stop_timesteps', default=500000, type=int, help='max training timesteps')
parser.add_argument('--stop_reward', default=0.8, type=int, help='stop reward')
args = parser.parse_args()


_TASK_OBSERVATIONS = ['RGBA_INTERLEAVED', 'reward', 'done']
PORT = 30051
# create worker
PCGWorker_ = PCGWorker(9,9)
# start from empty aera
wave_seed = PCGWorker_.build_wave()
_SEED = np.ones((81,1,2)).astype(np.int32)
# inital connectiy from probility space
mask , _ = PCGWorker_.connectivity_analysis(wave = wave_seed, visualize_ = False, to_file = False)
# reduce mask to 9x9 for processing
reduced_map = einops.reduce(mask,"(h a) (w b) -> h w", a=20, b=20, reduction='max').reshape(-1)
# use maxium playable area as probility space
_SPACE = np.flatnonzero(reduced_map == np.argmax(np.bincount(reduced_map))).astype(np.int32)

stop_iters = 50
stop_timesteps = 100000
stop_reward = 0.1

stop = {
        "training_iteration": stop_iters,
        "timesteps_total": stop_timesteps,
        # "episode_reward_mean": stop_reward,
}


os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"
# # try:
#     ray.init()


def env_creator(config):
    settings = EnvironmentSettings(create_world_settings={"seed": config["seed"]},join_world_settings={
                "agent_pos_space": config["agent_pos_space"],
                "object_pos_space":  config["object_pos_space"]}, timescale=config["timescale"])
    dm_env, port = load_from_disk(args.game, settings)
    gym_env = GymFromDMEnv(dm_env)
    return gym_env


from ray import tune
tune.register_env(
    "rpc_unity3d",
    lambda config: env_creator(config), 
)


if __name__ == "__main__":
    try:
        ray.init()
        config = {
                "env": "rpc_unity3d",
                "env_config": {
                    "seed": _SEED,
                    "agent_pos_space": _SPACE,
                    "object_pos_space": _SPACE,
                    "OBSERVATIONS": _TASK_OBSERVATIONS,
                    "timescale": args.timescale          # Unity时间默认2倍速
                },
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                # "model": {
                #     "custom_model": "my_model",
                #     "vf_share_layers": True,
                # },
                "num_workers": args.workers,  # parallelism
                "framework": "torch",
        }
        # stop_iters = 50
        # stop_timesteps = 100000
        stop_timesteps = args.stop_timesteps
        # stop_reward = 0.1

        stop = {
                # "training_iteration": stop_iters,
                "timesteps_total": stop_timesteps,
                # "episode_reward_mean": stop_reward,
        }
        if args.algrithm == "IMPALA":
            alg_config = impala.DEFAULT_CONFIG.copy()
        elif args.algrithm == "PPO":
            alg_config = ppo.DEFAULT_CONFIG.copy()
        elif args.algrithm == "A3C":
            alg_config = a3c.DEFAULT_CONFIG.copy()
        elif args.algrithm == "DQN":
            alg_config = dqn.DEFAULT_CONFIG.copy()
        alg_config.update(config)
        alg_config["num_gpus"] = 1
        alg_config["num_workers"] = args.workers
        alg_config["train_batch_size"] = args.train_batch_size
        alg_config["rollout_fragment_length"] = args.rollout_fragment_length
        alg_config["framework"] = "torch"
        alg_config["log_level"] = "DEBUG"
        alg_config["horizon"] = 2000
        alg_config["lr"] = 1e-3
        # alg_config["create_env_on_driver"] = True
        # trainer = ppo.PPOTrainer(config=ppo_config)
        # tune train
        tune.run(f"{args.algrithm}", config=alg_config, stop=stop)
        # Can optionally call trainer.restore(path) to load a checkpoint.

    except RuntimeError:
        ray.shutdown()
        ray.init()
    finally:
        # trainer.workers.stop()
        ray.shutdown()

