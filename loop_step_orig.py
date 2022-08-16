import wave
import numpy as np

from google.rpc import code_pb2
from google.rpc import status_pb2
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import spec_manager
from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils
import grpc
from time import sleep
import numpy as np
import time
import argparse
parser = argparse.ArgumentParser(description='loop test')
parser.add_argument('--ports', dest='port', default=30051, type=int)
args = parser.parse_args()
import pygame

_FRAMES_PER_SEC = 50
_FRAME_DELAY_MS = int(1000.0 // _FRAMES_PER_SEC)

_ACTION_NOTHING = 0
_ACTION_LOOKLEFT = 1
_ACTION_LOOKRIGHT = 2
_ACTION_LOOKUP = 3
_ACTION_LOOKDOWN = 4
_ACTION_LEFT = 5
_ACTION_RIGHT = 6
_ACTION_FORWARD = 7
_ACTION_BACKWARD = 8

_ACTION_PADDLE = 'paddle'
_ACTION_JUMP = 'jump'

_OBSERVATION_CAMERA = 'RGBA_INTERLEAVED'
_SPACE = np.array([ 1,  2,  4,  5,  6,  9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22,
       23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
       40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58,
       59, 65, 66, 67, 68, 77, 78], dtype=np.int32)
_TEMP = np.ones((81,1,2)).astype(np.int32)

# 随便生成一张地图及连通域
import sys 
sys.path.append("..") 
from pcgworker.PCGWorker import *
import einops
PCGWorker_ = PCGWorker(9,9)
wave_seed = PCGWorker_.generate()
mask , _ = PCGWorker_.connectivity_analysis(wave = wave_seed,visualize_ = False, to_file = False)
test_reduce = einops.reduce(mask,"(h a) (w b) -> h w", a=20, b=20, reduction='max')
reduced_map = test_reduce.reshape(-1)
space = np.flatnonzero(reduced_map == np.argmax(np.bincount(reduced_map))).astype(np.int32)
_TEMP = np.array(wave_seed.wave_oriented).astype(np.int32)
_SPACE = space

# _OBSERVATION_CAMERA = 'Camera'
def main():
    # pygame.init()

    port = args.port
    timeout = 10
    
    with grpc.insecure_channel(f'localhost:{port}') as channel:
        grpc.channel_ready_future(channel).result(timeout)
        connection = dm_env_rpc_connection.Connection(channel)
        print(connection)
        print("create and join world")
        env, world_name = dm_env_adaptor.create_and_join_world(
        connection, create_world_settings={}, join_world_settings={})
        print("joined world:", world_name)
        count =0
        with env:
            start = time.time()
            while count<10000:
                actions = {_ACTION_PADDLE: [7],
                           _ACTION_JUMP: [1]}
                timestep = env.step(actions)
                count += 1
            print("time:", time.time() - start)
        connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=world_name))

if __name__ == "__main__":
    main()
