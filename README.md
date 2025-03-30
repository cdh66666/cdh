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
- [CUDA 12.0](https://blog.51cto.com/u_16213611/10480090)
- Python 3.7 
- [PyTorch 1.10.1+cu111](https://pytorch.org/get-started/previous-versions/)
- Isaac Gym：preview4

 
**1. 创建环境&安装pytorch**
- `conda create -n cdh python=3.7.16`
- `conda activate cdh`
- `pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html`

**2. 安装Isaac Gym**

-  从 [Isaac Gym官网](https://developer.nvidia.com/isaac-gym) 下载 Isaac Gym preview4 。
-  解压至主目录：
    ```bash
    tar -xzvf IsaacGym_Preview_4_Package.tar.gz -C ~
    ```
-  开始安装环境：
    ```bash
    cd isaacgym/python && pip install -e .
    ```
-  设置快速激活环境的别名，输入 `cdh` 即可激活虚拟环境：
    ```bash
    echo "alias cdh='conda activate cdh'" >> ~/.bashrc && source ~/.bashrc
    ```

**3. 克隆仓库**
```bash
cd ~ && git clone https://github.com/cdh66666/cdh.git && cd cdh
```

**4. 测试示例**
```bash
cdh && cd ~/cdh && python test.py
```



<!-- **4. 安装cdh**
```bash
cdh && cd ~/cdh && pip install -e . ##待定
``` -->

### 教程
**官方文档**：解压安装包后，可以在`isaacgym/docs`目录下找到`index.html`文件，双击即可打开官方文档。
1. 训练一个策略：
  - `cdh && cd ~/cdh/scripts && python train.py`
 

2. 运行并导出最新的策略：
  - `cdh && cd ~/cdh/scripts && python play.py`
 
 