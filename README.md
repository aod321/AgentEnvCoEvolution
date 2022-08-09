# 交互原理示意图

![示意图](https://cloud.tsinghua.edu.cn/f/718a8682c063447e979b/?dl=1)

# 协同演化过程演示

<video src="https://cloud.tsinghua.edu.cn/f/0319ceed3f214085ba1f/?dl=1" controls="controls" width="500" height="300"></video>

# 多世界支持

[Gitee Wiki](https://gitee.com/aod321/AgentEnvCoEvolution/wikis/%E5%A4%9A%E4%B8%96%E7%95%8C%E6%94%AF%E6%8C%81)

[THBI Git WIKI](http://git.thbi.cc/yinzi/AgentEnvCoEvolution/wiki/%E5%A4%9A%E4%B8%96%E7%95%8C%E6%94%AF%E6%8C%81)

# 安装

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

# 键盘控制

1. 开启Unity工程TIleMapRender，运行，运行成功会提示监听端口
2. 运行test_grpc.py
3. 鼠标焦点放在pygame窗口中，然后使用键盘控制| 按键 | 动作   |
   | ---- | ------ |
   | ↑   | 向前走 |
   | ↓   | 向后走 |
   | ←   | 向左走 |
   | →   | 向右走 |
   | Q    | 向左转 |
   | E    | 向右转 |
   | R    | 向上转 |
   | F    | 向下转 |
   | 空格 | 跳跃   |

# agent和环境联合演化

1. 依赖李今的pcgworker库，该库目前需手动安装，并注意路径

   https://gitee.com/electricsoul/pcgworker
2. 打开Unity工程TIleMapRender，运行，运行成功会提示监听端口
3. 训练

   ```shell
   python simple_train.py --train_eposides 2000 --train_steps 5000 --evlaute_steps 2000 --evol_evaluate_steps 2000 --evlaute_eposide 10
   ```
