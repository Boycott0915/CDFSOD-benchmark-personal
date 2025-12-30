# #!/bin/bash
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# shot_list=(5)

# for shot in "${shot_list[@]}"; do
#   echo "------------------------------------------------"
#   echo ">>> 开始 DIOR ${shot}-shot 的微调训练..."
#   echo "------------------------------------------------"

#   # # 【修复点】在这里也加上 DE.OFFLINE_RPN_CONFIG 参数
#   # CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#   #   --num-gpus 4 \
#   #   --config-file configs/dior/vitb_shot${shot}_dior_finetune.yaml \
#   #   OUTPUT_DIR output/vitb/dior_${shot}shot/ \
#   #   DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
#   #   SOLVER.IMS_PER_BATCH 4 \
#   #   SOLVER.BASE_LR 0.0001 2>&1 | tee output/vitb/dior_${shot}shot/train_log.txt

#   # echo ">>> DIOR ${shot}-shot 训练完成！立刻进行推理评价..."

#   CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/train_net.py \
#     --num-gpus 4 \
#     --eval-only \
#     --config-file configs/dior/vitb_shot${shot}_dior_finetune.yaml \
#     MODEL.WEIGHTS output/vitb/dior_${shot}shot/model_final.pth \
#     DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
#     OUTPUT_DIR output/vitb/dior_${shot}shot/ \
#     SOLVER.IMS_PER_BATCH 4 \
#     INPUT.MIN_SIZE_TEST 600 2>&1 | tee output/vitb/dior_${shot}shot/eval_fix_log.txt
# done


datalist=(
dior
)
shot_list=(
10
)
model_list=(
#"l"
"b"
#"s"
)
for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do
      CUDA_VISIBLE_DEVICES=6,7 python tools/train_net.py --num-gpus 2 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS weights/trained/few-shot/vit${model}_0089999.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/vit${model}/${dataset}_${shot}shot/
    done
  done
done