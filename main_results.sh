#!/bin/bash
# 1. 显存优化：碎片整理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 2. 针对 1, 5, 10 shot 依次进行训练 (Finetune)
shot_list=(1 5 10)

for shot in "${shot_list[@]}"; do
  echo "------------------------------------------------"
  echo ">>> 开始 DIOR ${shot}-shot 的微调训练..."
  echo "------------------------------------------------"

  # 【核心修改点】
  # SOLVER.IMS_PER_BATCH: 如果还报 OOM，就把 4 改成 2
  # SOLVER.BASE_LR: 1-shot 建议 0.0001, 5/10-shot 建议 0.0002
  CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
    --num-gpus 4 \
    --config-file configs/dior/vitb_shot${shot}_dior_finetune.yaml \
    OUTPUT_DIR output/vitb/dior_${shot}shot/ \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.0001 2>&1 | tee output/vitb/dior_${shot}shot/train_log.txt

  echo ">>> DIOR ${shot}-shot 训练完成！立刻进行推理评价..."

  # 训练完立刻接上推理，不用手动再跑
  CUDA_VISIBLE_DEVICES=4,5,6,7d python tools/train_net.py \
    --num-gpus 4 \
    --eval-only \
    --config-file configs/dior/vitb_shot${shot}_dior_finetune.yaml \
    MODEL.WEIGHTS output/vitb/dior_${shot}shot/model_final.pth \
    DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
    OUTPUT_DIR output/vitb/dior_${shot}shot/ \
    SOLVER.IMS_PER_BATCH 4 \
    INPUT.MIN_SIZE_TEST 600 2>&1 | tee output/vitb/dior_${shot}shot/eval_fix_log.txt
done