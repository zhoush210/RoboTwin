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
# 修改policy/pi0/src/openpi/training/config.py中的checkpoint_base_dir（ckpt的保存地址）
bash finetune.sh pi0_base_aloha_robotwin_full demo_clean 0,1,2,3
# 推理
# 修改policy/pi0/deploy_policy.yml中的配置，尤其是ckpt
bash eval.sh beat_block_hammer demo_clean pi0_base_aloha_robotwin_full demo_clean 0 0
```

## openvla
```bash
cd policy/openvla-oft
# 处理数据
python preprocess_aloha.py \
  --dataset_path /mnt/nvme1/shihuiz/robotwin/blocks_ranking_rgb/demo_clean/data \
  --out_base_dir /mnt/nvme1/shihuiz/robotwin/blocks_ranking_rgb/processed_openvla/ \
  --percent_val 0.05 \
  --instruction_dir /mnt/nvme1/shihuiz/robotwin/blocks_ranking_rgb/demo_clean/instructions/
# 创建 blocks_ranking_rgb_builder，在 configs.py、transforms.py、mixtures.py 为其添加条目，将数据集注册到数据加载器。
# 验证：
python -m datasets.blocks_ranking_rgb_builder
```
