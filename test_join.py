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
_WORLD_NAME = "World_0"

# _OBSERVATION_CAMERA = 'Camera'

def main():
    pygame.init()

    port = 30051
    timeout = 10
    
    with grpc.insecure_channel(f'localhost:{port}') as channel:
        grpc.channel_ready_future(channel).result(timeout)
        connection = dm_env_rpc_connection.Connection(channel)
        print(connection)

        print(f"join world: {_WORLD_NAME}")
        env = dm_env_adaptor.join_world(connection, world_name=_WORLD_NAME, join_world_settings={
            "agent_pos_space": _SPACE,
            "object_pos_space": _SPACE
        })
        print("joined world:", _WORLD_NAME)

        window_surface = pygame.display.set_mode((84,84), 0, 32)
        pygame.display.set_caption(_WORLD_NAME)
        window_surface.fill((128, 128, 128))
        with env:
            keep_running = True
            while keep_running:
                requested_action = _ACTION_NOTHING
                is_jumping = 0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        keep_running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            requested_action = _ACTION_LEFT
                        elif event.key == pygame.K_RIGHT:
                            requested_action = _ACTION_RIGHT
                        if event.key == pygame.K_UP:
                            requested_action = _ACTION_FORWARD
                        elif event.key == pygame.K_DOWN:
                            requested_action = _ACTION_BACKWARD
                        elif event.key == pygame.K_q:
                            requested_action = _ACTION_LOOKLEFT
                        elif event.key == pygame.K_e:
                            requested_action = _ACTION_LOOKRIGHT
                        elif event.key == pygame.K_r:
                            requested_action = _ACTION_LOOKUP
                        elif event.key == pygame.K_f:
                            requested_action = _ACTION_LOOKDOWN
                        elif event.key == pygame.K_SPACE:
                            is_jumping = 1
                        elif event.key == pygame.K_ESCAPE:
                            keep_running = False
                            break

                actions = {_ACTION_PADDLE: [requested_action],
                           _ACTION_JUMP: [is_jumping]}
                timestep = env.step(actions)
                # print(timestep)
                # if timestep.reward != 0:
                    # print("reward:", timestep.reward)
                if(timestep.reward):
                    print("reward: ", timestep.reward)
                # print(dir(image))
                # print(image[:10])
                pixels = timestep.observation[_OBSERVATION_CAMERA]
                image = pygame.image.frombuffer(pixels, (84, 84), 'RGBA')
                image = pygame.transform.flip(image, False, True)
                window_surface.blit(image, (0, 0))
                pygame.display.update()
                # pygame.time.wait(_FRAME_DELAY_MS)
            
        # connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=world_name))

if __name__ == "__main__":
    main()
