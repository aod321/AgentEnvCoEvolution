# Framework

![示意图](https://cloud.tsinghua.edu.cn/f/718a8682c063447e979b/?dl=1)

# Unity3D dm_env_rpc Server implementation
Due to the large size of the complete project file, only the Scripts are kept.

[GRPCServer repo](https://github.com/aod321/GRPCServer)
# Agent-world co-evolution

<!-- this link magically rendered as video, unfortunately not in docs -->


<!--
<a href='https://cloud.tsinghua.edu.cn/f/0319ceed3f214085ba1f/?dl=1' >
<div align="center">
  <img src="http://arogozhnikov.github.io/images/einops/einops_video.gif" alt="einops package examples" />
  <br>
  <small><a href='https://cloud.tsinghua.edu.cn/f/0319ceed3f214085ba1f/?dl=1'>This video in high quality (mp4)</a></small>
  <br><br>
</div>
</a>
-->
![](https://cloud.tsinghua.edu.cn/f/efdd19d0ed444bccaa98/?dl=1)
![](https://cloud.tsinghua.edu.cn/f/bf6468c7360b41e99653/?dl=1)
# Multi-world and Multi-agent 

## Demos

1. multi-agent
![multi-agent](https://cloud.tsinghua.edu.cn/f/d3a60a1b4c8e417a8572/?dl=1)

2. multi-world and multi-agent
![multi-world&multi-agent](https://cloud.tsinghua.edu.cn/f/28b5e72dd2fd4878901e/?dl=1)

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
