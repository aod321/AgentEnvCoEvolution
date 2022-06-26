from dm_env_rpc.v1 import dm_env_adaptor
import AgentEnvCoEvolution._load_environment as dm_tasks
from AgentEnvCoEvolution.gym_wrapper import GymFromDMEnv
import numpy as np
import einops
from pcgworker.PCGWorker import *
import matplotlib.pyplot as plt
import os
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import os
from AgentEnvCoEvolution._load_environment import EnvironmentSettings, load_from_disk
import argparse
from ray import tune

# workaround for issues after 1.43.0, should be fixed in next grpcio version
# https://github.com/ray-project/ray/issues/22518
os.environ["GRPC_FORK_SUPPORT_ENABLED"] = "1"

# argparser
parser = argparse.ArgumentParser(description='RLlib multi-thread sampling Unity3D envs example')
parser.add_argument('--game', default="/Users/yinzi/Downloads/test/m1_map_rpc_built_x64.app/Contents/MacOS/tilemap_render", type=str, help='path to built unity runtime')
parser.add_argument('--workers', default=4, type=int, help='workers number: 1, 4, 8, 16')
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


try:
    ray.init()
except RuntimeError:
    ray.shutdown()
    ray.init()


def env_creator(config):
    settings = EnvironmentSettings(create_world_settings={"seed": config["seed"]},join_world_settings={
                "agent_pos_space": config["agent_pos_space"],
                "object_pos_space":  config["object_pos_space"]}, timescale=config["timescale"])
    dm_env = load_from_disk(args.game, settings)
    gym_env = GymFromDMEnv(dm_env)
    return gym_env


tune.register_env(
    "rpc_unity3d",
    lambda config: env_creator(config), 
)

config = {
        "env": "rpc_unity3d",
        "env_config": {
             "seed": _SEED,
            "agent_pos_space": _SPACE,
            "object_pos_space": _SPACE,
            "OBSERVATIONS": _TASK_OBSERVATIONS,
            "timescale": 2          # Unity时间默认2倍速
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # "model": {
        #     "custom_model": "my_model",
        #     "vf_share_layers": True,
        # },
        "num_workers": 0,  # parallelism
        "framework": "torch",
 }

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)
ppo_config["num_gpus"] = 0
ppo_config["num_workers"] = args.workers
ppo_config["train_batch_size"] = 8000
ppo_config["rollout_fragment_length"] = 2000
ppo_config["framework"] = "torch"
ppo_config["lr"] = 1e-3
# ppo_config["create_env_on_driver"] = True
trainer = ppo.PPOTrainer(config=ppo_config)
# print(trainer.workers.foreach_env(lambda env: env.sample()))s
import time
T1 = time.time()
sample_batches = ray.get(
            [worker.sample.remote() for worker in trainer.workers.remote_workers()]
        )
print(len(sample_batches))
print(sample_batches[0]['obs'].shape)
T2 =time.time()
print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    # print(trainer.workers.local_worker)
    # # run manual training loop and print results after each iteration
    # for i in range(stop_iters):
    #     result = trainer.train()
    #     print(pretty_print(result))
    #     if i % 100 == 0:
    #         checkpoint = trainer.save()
    #     print("checkpoint saved at", checkpoint)
    #     # # stop training of the target train steps or reward are reached
    #     # if (
    #     #     result["timesteps_total"] >= stop_timesteps
    #     # ):
        #     break