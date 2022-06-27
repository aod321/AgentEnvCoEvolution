import os
import ray
import argparse
import logging
import logging
from WFCUnity3DEnv import WFCUnity3DEnv
from ray import tune
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
import logging
# import random
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.impala as impala
import ray.rllib.agents.dqn as dqn
import sys
sys.path.append('../')
from pcgworker.PCGWorker import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


PCGWorker_ = PCGWorker(9,9)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--game",
    type=str,
    default="/Users/yinzi/Downloads/test/m1_map_rpc_built_x64.app/Contents/MacOS/tilemap_render",
    help="The path of the Unity3D built",
)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Trainer state.",
)
parser.add_argument(
    "--stop_iters", type=int, default=1000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop_timesteps", type=int, default=1000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop_reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=500,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument('--algrithm', default="IMPALA", type=str, help='algrithm: IMPALA, A3C, PPO, DQN')
parser.add_argument("--train_workers", type=int, default=4)
parser.add_argument("--eval_workers", type=int, default=4)
parser.add_argument('--timescale', default=2, type=int, help='timescale for unity3d timesys')
parser.add_argument('--train_batch_size', default=2048, type=int, help='batch_size for training')
parser.add_argument('--rollout_fragment_length', default=512, type=int, help='batch_size for training')

def env_creator(config):
    rllib_env = WFCUnity3DEnv(config=config, wfc_size=9)
    return rllib_env

def custom_eval_function(trainer, eval_workers):
    """a custom evaluation function.
    Args:
        trainer: trainer class to evaluate.
        eval_workers: evaluation workers.
    Returns:
        metrics: evaluation metrics dict.
    """

    for i in range(5):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(
        remote_workers=eval_workers.remote_workers(), timeout_seconds=99999
    )
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())
    logger.debug(f"evaluation_metrics: {metrics}")
    """After-evaluation callback."""
    def should_continue(results):
        print(results)
        eposides_reward_mean = metrics["episode_reward_mean"]
        # # nonzero_episodes = (eposides_reward > 0).sum()
        # # Map is too easy since there are too many succussful episodes 
        return eposides_reward_mean > 0.5
    count = 0
        # keep current wave in trainning workers
    current_wave = trainer.workers.foreach_env(lambda env: env.wave)[1][0]
    mutated_wave = current_wave
    # is Time to evolute a new map            
    while should_continue(metrics):
        count += 1
        logger.info(f"Start Evolving map for the {count} time")
         # mutate and render the new map in evalutation workers
        mutated_wave = PCGWorker_.mutate(mutated_wave, 81)
        def fn(env):
            env.set_wave(mutated_wave)
            env.render_in_unity()
        eval_workers.foreach_env(fn)
        logger.info(f"Evaluating the evolved map now")
        for i in range(5):
            print("Custom evaluation round", i)
            # Calling .sample() runs exactly one episode per worker due to how the
            # eval workers are configured.
            ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
            # Collect the accumulated episodes on the workers, and then summarize the
            # episode stats into a metrics dict.
            episodes, _ = collect_episodes(
                remote_workers=eval_workers.remote_workers(), timeout_seconds=99999
            )
            metrics = summarize_episodes(episodes)
    logger.info("Got a New Map!")
    logger.debug(f"New Map Evaluation metrics: {metrics}")
    # Rendering the new map for training_iteration
    def fn(env):
        env.render_in_unity()
    trainer.workers.foreach_env(fn)
    return metrics


if __name__ == "__main__":
    try:
        ray.init()
    except:
        ray.shutdown()
    try:
        args = parser.parse_args()

        tune.register_env(
            "rpc_unity3d",
            lambda config: env_creator(config), 
        )

        config = {
                "env": "rpc_unity3d",
                "env_config": {
                    "filename": args.game,
                    "timescale": args.timescale # Unity时间默认2倍速
                },
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                # "model": {
                #     "custom_model": "my_model",
                #     "vf_share_layers": True,
                # },
                "train_batch_size": args.train_batch_size,
                "rollout_fragment_length": args.rollout_fragment_length,
                "num_workers": args.train_workers,  # parallelism
                "horizon": args.horizon,
                "framework": "torch",
                "evaluation_num_workers": args.eval_workers,
                "custom_eval_function": custom_eval_function,
                # Enable evaluation, once per training iteration.
                "evaluation_interval": 1,
                # Run 10 episodes each time evaluation runs.
                "evaluation_duration": 10,  
                "create_env_on_driver": False,
                "explore": True,
                "exploration_config":{
                    "type": "EpsilonGreedy",
                },
                "log_level": "DEBUG"
        }
        stop = {
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            # "episode_reward_mean": args.stop_reward,
        }
        if args.algrithm == "IMPALA":
            alg_config = impala.DEFAULT_CONFIG.copy()
        elif args.algrithm == "PPO":
            alg_config = ppo.DEFAULT_CONFIG.copy()
        elif args.algrithm == "DQN":
            alg_config = dqn.DEFAULT_CONFIG.copy()
        elif args.algrithm == "A3C":
            alg_config = a3c.DEFAULT_CONFIG.copy()
        alg_config.update(config)
        # Run the experiment.
        results = tune.run(
            args.algrithm,
            config=config,
            stop=stop,
            verbose=1,
            checkpoint_freq=5,
            checkpoint_at_end=True,
            restore=args.from_checkpoint,
        )
    finally:
        ray.shutdown()
