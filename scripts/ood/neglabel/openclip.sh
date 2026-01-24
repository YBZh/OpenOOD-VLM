

####################################### OOD datasets under OpenOOD setting.
tau=1.0
beta=1.0
######################### clip + ood + NegLabel
prompt=nice
random_permute=True
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
for in_score in sum
do
    for group_num in 100
    do
        python main.py \
        --config configs/datasets/imagenet/imagenet_train_ood.yml \
        configs/networks/fixed_clip.yml \
        configs/pipelines/test/test_fsood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/mcm.yml \
        --dataset.train.batch_size 128 \
        --dataset.train.few_shot 0 \
        --dataset.num_classes 11000 \
        --evaluator.name ood_clip \
        --network.name fixedclip_negoodprompt_openclip \
        --network.backbone.ood_number 10000 \
        --network.backbone.text_prompt ${prompt} \
        --network.backbone.text_center True \
        --network.pretrained False \
        --postprocessor.APS_mode False \
        --postprocessor.name oneoodpromptdevelop \
        --postprocessor.postprocessor_args.group_num ${group_num}  \
        --postprocessor.postprocessor_args.random_permute ${random_permute}  \
        --postprocessor.postprocessor_args.tau ${tau}  \
        --postprocessor.postprocessor_args.beta ${beta}  \
        --postprocessor.postprocessor_args.in_score ${in_score}  \
        --num_gpus 1 --num_workers 6 \
        --merge_option merge \
        --output_dir ./debug_neglabel_official/ \
        --mark ${in_score}_beta${beta}_neg10k_group_num_${group_num}_random_${random_permute}_official
    done
done

# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:20<00:00,  2.20it/s]
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:50<00:00,  3.81it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 69.20, AUROC: 77.40 AUPR_IN: 76.45, AUPR_OUT: 77.49
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:15<00:00,  1.46it/s]
# Computing metrics on ninco dataset...
# FPR@95: 64.13, AUROC: 80.46 AUPR_IN: 96.68, AUPR_OUT: 42.81
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 66.67, AUROC: 78.93 AUPR_IN: 86.56, AUPR_OUT: 60.15
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:33<00:00,  1.19it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 4.10, AUROC: 99.13 AUPR_IN: 99.79, AUPR_OUT: 97.06
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:10<00:00,  1.94it/s]
# Computing metrics on textures dataset...
# FPR@95: 42.78, AUROC: 89.71 AUPR_IN: 98.59, AUPR_OUT: 58.72
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:24<00:00,  2.53it/s]
# Computing metrics on openimageo dataset...
# FPR@95: 34.58, AUROC: 93.07 AUPR_IN: 97.17, AUPR_OUT: 86.24
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 27.15, AUROC: 93.97 AUPR_IN: 98.52, AUPR_OUT: 80.68
# ACC: 68.61
# ──────────────────────────────────────────────────────────────────────