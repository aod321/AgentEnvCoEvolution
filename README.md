# Framework

![示意图](https://cloud.tsinghua.edu.cn/f/718a8682c063447e979b/?dl=1)

# Unity3D dm_env_rpc Server implementation
Due to the large size of the complete project file, only the Scripts are kept.

[GRPCServer repo](https://github.com/aod321/GRPCServer)
# Agent-world co-evolution

![](https://cloud.tsinghua.edu.cn/f/efdd19d0ed444bccaa98/?dl=1)
![](https://cloud.tsinghua.edu.cn/f/bf6468c7360b41e99653/?dl=1)
# Multi-world and Multi-agent 

## Demos

1. multi-agent

<img src="imgs/third.gif?raw=true" width="800px">

2. multi-world and multi-agent

<img src="imgs/fourth.gif?raw=true" width="800px">


## Development Diary(in Chinese)
[Gitee Wiki](https://gitee.com/aod321/AgentEnvCoEvolution/wikis/%E5%A4%9A%E4%B8%96%E7%95%8C%E6%94%AF%E6%8C%81)

[THBI Git WIKI](http://git.thbi.cc/yinzi/AgentEnvCoEvolution/wiki/%E5%A4%9A%E4%B8%96%E7%95%8C%E6%94%AF%E6%8C%81)

# Installation

```shell
git clone http://git.thbi.cc/yinzi/AgentEnvCoEvolution.git
conda create --name grpcunity python=3.9
conda activate grpcunity
pip install dm_env_rpc
conda install grpcio
pip install docker
pip install portpicker
pip install stable-baselines3
pip install einops
pip install opencv-python
pip install colorama
pip install pygame
pip install tqdm
```

# Keyboard Control

1. Open the Unity project TileMapRender, run it, and it will prompt to listen to the gRPC port if it runs successfully
2. python3 test_grpc.py
3. Mouse focus in the pygame window, then use the keyboard to control:
   | Keys | Actions   |
   | ---- | ------ |
   | ↑   | Up |
   | ↓   | Down |
   | ←   | Left |
   | →   | Right |
   | Q    | Turn Left |
   | E    | Turn Right  |
   | R    | Look Up |
   | F    | Look Down |
   | Space | Jump   |

# Joint WFC mutation

1.  Install WFC repo
   https://gitee.com/electricsoul/pcgworker
2.  Open the Unity project TileMapRender, run it, and it will prompt to listen to the gRPC port if it runs successfully
3. Start Training

   ```shell
   python simple_train.py --train_eposides 2000 --train_steps 5000 --evlaute_steps 2000 --evol_evaluate_steps 2000 --evlaute_eposide 10
   ```
