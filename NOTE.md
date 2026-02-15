## 文档
https://robotwin-platform.github.io/doc/usage/index.html

## 安装
- ubuntu24要使用cuda12.4
```bash
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
bash script/_install.sh
conda install -c conda-forge ffmpeg

cd policy/MemoryVLA
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
```
## 生成数据
- 修改`task_config/demo_clean.yml`中的`save_path`
```bash
bash collect_data.sh blocks_ranking_rgb demo_clean 0
```

## ACT
```bash
cd policy/ACT
# 处理数据
bash process_data.sh blocks_ranking_rgb demo_clean 50
```

## pi0
```bash
cd policy/pi0
# 处理数据
bash process_data_pi0.sh blocks_ranking_rgb demo_clean 50
# repo_id: demo_clean
bash generate.sh /mnt/nvme1/shihuiz/robotwin/processed_data/ demo_clean
# 统计数据
uv run scripts/compute_norm_stats.py --config-name pi0_base_aloha_robotwin_full
# 训练
# 修改policy/pi0/src/openpi/training/config.py中的 checkpoint_base_dir （ckpt的保存地址）
bash finetune.sh pi0_base_aloha_robotwin_full demo_clean 0,1,2,3
# 推理
# 修改policy/pi0/deploy_policy.yml中的配置，尤其是ckpt
bash eval.sh blocks_ranking_rgb demo_clean pi0_base_aloha_robotwin_full demo_clean 0 0
```

## pi05
```bash
cd policy/pi05
# 处理数据
bash process_data_pi05.sh blocks_ranking_rgb demo_clean 50
# repo_id: demo_clean
bash generate.sh /mnt/nvme1/shihuiz/robotwin/processed_data/ demo_clean
# 统计数据
uv run scripts/compute_norm_stats.py --config-name pi05_aloha_full_base
# 训练
# 修改policy/pi0/src/openpi/training/config.py中的 checkpoint_base_dir （ckpt的保存地址）
bash finetune.sh pi05_aloha_full_base demo_clean 0,1,2,3
# 推理
# 修改policy/pi05/deploy_policy.yml中的配置，尤其是ckpt
conda activate RoboTwin && source .venv/bin/activate
bash eval.sh blocks_ranking_rgb demo_clean pi05_aloha_full_base demo_clean 0 0
```