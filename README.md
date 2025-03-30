## 来自cdh的基于学习的运动控制
本仓库包含cdh的运动控制研究。
## 🔥 最新消息
- [2025年3月30日] cdh创建了本仓库。
## 📝 待办事项列表
- [x] 完成本仓库关于环境的搭建并给出搭建流程。
- [ ] 完成本仓库关于机器人模型的搭建并给出搭建流程。
- [ ] 完成本仓库关于策略的搭建并给出搭建流程。

## 📚 入门指南

### 安装

在以下环境中测试了代码：

- Ubuntu 20.04
- NVIDIA驱动程序：535.183.01
- CUDA 11.1
- Python 3.7 
- PyTorch 1.8.1+cu111
- Isaac Gym：preview4

 
#### 1. 创建环境
1. 从 [Isaac Gym官网](https://developer.nvidia.com/isaac-gym) 下载 Isaac Gym：preview4 并解压至主目录：
    ```bash
    tar -xzvf IsaacGym_Preview_4_Package.tar.gz -C ~
    ```
2. 将环境名字从 `rlgpu` 替换为 `cdh`：
    ```bash
    sed -i 's/rlgpu/cdh/g' ~/isaacgym/create_conda_env_rlgpu.sh
    sed -i 's/rlgpu/cdh/g' ~/isaacgym/python/rlgpu_conda_env.yml
    ```
3. 开始创建环境：
    ```bash
    cd ~/isaacgym && ./create_conda_env_rlgpu.sh
    ```
4. 设置快速激活环境的别名，输入 `cdh` 即可激活虚拟环境：
    ```bash
    echo "alias cdh='conda activate cdh'" >> ~/.bashrc && source ~/.bashrc
    ```

#### 2. 克隆仓库
```bash
git clone https://github.com/cdh66666/cdh.git
cd ~/cdh
```


### 教程

1. 训练一个策略：

  - `cd ~/cdh/scripts`
  - `python train.py`

2. 运行并导出最新的策略：
  - `cd ~/cdh/scripts`
  - `python play.py`

 