from gym_wrapper import GymFromDMEnv
import sys
sys.path.append('../')
from pcgworker.PCGWorker import *
import matplotlib.pyplot as plt
import dm_env
import _load_environment as dm_tasks
import einops
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import tensor_utils
from dm_env_rpc.v1 import dm_env_rpc_pb2


def dm_env_creator_from_port(config, port):
    Unity_connection_details = dm_tasks._connect_to_environment(port, 
                    create_world_settings={"seed": config["seed"]},
                    join_world_settings={
                                        "agent_pos_space": config["agent_pos_space"],
                                        "object_pos_space": config["object_pos_space"]
                                        }
                    )
    dm_env = dm_tasks._DemoTasksProcessEnv(Unity_connection_details, config["OBSERVATIONS"], num_action_repeats=config["num_action_repeats"])
    return dm_env

def dm_env_creator_from_local_disk(config):
    settings = dm_tasks.EnvironmentSettings(create_world_settings={"seed": config["seed"]},join_world_settings={
                "agent_pos_space": config["agent_pos_space"],
                "object_pos_space":  config["object_pos_space"]}, timescale=config["timescale"])
    dm_env = dm_tasks.load_from_disk(config["filename"], settings)
    return dm_env


# Add WFC and gRPC support for unity3D RLLib enviroment
class WFCUnity3DEnv(GymFromDMEnv):
    def __init__(self, env: dm_env.Environment=None, wfc_size=9, config=None, file_name=None, port=30051):
        self.world_name = None
        self.height_map = None
        # create worker
        self.PCGWorker_ = PCGWorker(wfc_size, wfc_size)
        # start from empty aera
        self.wave = self.PCGWorker_.build_wave()
        self.TASK_OBSERVATIONS = ['RGBA_INTERLEAVED', 'reward', 'done']
        self._SPACE = self.get_space_from_wave(self.wave)
        # all empty tile
        self._SEED = np.ones((wfc_size * wfc_size,1,2)).astype(np.int32)
        self.port = port
        if config is None:
            config = {
                "seed": self._SEED,
                "agent_pos_space": self._SPACE,
                "object_pos_space": self._SPACE,
                "OBSERVATIONS": self.TASK_OBSERVATIONS,
                "num_action_repeats": 1,
                "filename": file_name,
                "timescale": 2
            }
        else:
            if "wave" in config:
                if config["wave"]:
                    self.set_wave(config["wave"])
            if "seed" not in config:
                config["seed"] = self._SEED
            if "agent_pos_space" not in config:
                config["agent_pos_space"] = self._SPACE
            if "object_pos_space" not in config:
                config["object_pos_space"] = self._SPACE
            if "OBSERVATIONS" not in config:
                config["OBSERVATIONS"] = self.TASK_OBSERVATIONS
            if "num_action_repeats" not in config:
                config["num_action_repeats"] = 1
            if "filename" not in config:
                config["filename"] = file_name
            if "timescale" not in config:
                config["timescale"] = 2
        
        if env is None:
            if config["filename"] is None:
                 # using port 30051 as default
                if self.port is None:
                    self.port = 30051
                env = dm_env_creator_from_port(config, self.port)
            else:
                env, self.port = dm_env_creator_from_local_disk(config)
        super().__init__(env)

    def get_space_from_wave(self, wave=None):
        if not wave:
            wave = self.wave
        mask , _ = self.PCGWorker_.connectivity_analysis(wave = wave, visualize_ = False, to_file = False)
        # reduce mask to 9x9 for processing
        reduced_map = einops.reduce(mask,"(h a) (w b) -> h w", a=20, b=20, reduction='max').reshape(-1)
        # use maxium playable area as probility space
        return np.flatnonzero(reduced_map == np.argmax(np.bincount(reduced_map))).astype(np.int32)

    def create_and_join_world(self):
        try:
            connection = dm_tasks._connect_to_environment(self.port, 
                                    create_world_settings={"seed": self._SEED},
                                    join_world_settings={
                                                        "agent_pos_space": self._SPACE,
                                                        "object_pos_space": self._SPACE
                                                        }
                                    )
            self.connection_details ,self.world_name = connection
            self._env = dm_tasks._DemoTasksProcessEnv(connection, self.TASK_OBSERVATIONS, num_action_repeats=1)
            
        except Exception as e:
                print("Recreate Unity Map World Failed")
                raise e
   
    def render_in_unity(self, map_seed=None):
        if map_seed is None:
            map_seed = self._SEED
            space = self._SPACE
        else:
            space = self.get_space_from_wave(map_seed)
        print("reset world and agent")
        self._connection.send(
            dm_env_rpc_pb2.ResetWorldRequest(
                world_name=self._world_name,
                settings={
                    "seed": tensor_utils.pack_tensor(map_seed)
                }))

        self._connection.send(
            dm_env_rpc_pb2.ResetRequest(
                settings={
                    "agent_pos_space": tensor_utils.pack_tensor(space),
                    "object_pos_space": tensor_utils.pack_tensor(space)
                }))
        self.reset()

    # conditoinal WFC mutation
    def mutate_a_new_map(self, base_wave=None, size=81):
        if base_wave is None:
            base_wave = self.wave
        self.wave = self.PCGWorker_.mutate(base_wave, size)
        self._SPACE = self.get_space_from_wave(self.wave)
        result_seed, success = self.wave.get_result()
        if success:
            self._SEED = np.array(result_seed).astype(np.int32)
        else:
            self._SEED = np.ones((81,1,2)).astype(np.int32)
        return self.wave
    
    def set_wave(self, wave):
        self.wave = wave
        self._SPACE = self.get_space_from_wave(wave)
        result_seed, success = wave.get_result()
        if success:
            self._SEED = np.array(result_seed).astype(np.int32)
        else:
            self._SEED = np.ones((81,1,2)).astype(np.int32)    
    
    def get_wave(self):
        return self.wave

