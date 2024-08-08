
##### test with pre-trained classifier weights
# mix_strategy=mixup
# mix_strategy=cutmix
# mix_strategy=manimix
# mix_strategy=geomix
# mix_strategy=manimix_wccm
# mix_strategy=geomix_wccm
# mix_strategy=mixup_manimix_wccm
# mix_strategy=mixup_geomix_wccm
mix_strategy=wccm

depth=5
alpha=1
beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
total_gs_num=1000
loss_components=textentropy
gs_loss_weight=1-1-1-1
OOD_NUM=10000
shot_num=16
weight_decay=0.
optimizer=sgd
# N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# --evaluator.name fsood_clip \
group_num=100
random_permute=True
for seed in 0 
do
    for lr in 0.1 
    do
        python main.py \
            --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
            configs/networks/coop.yml \
            configs/pipelines/test/test_ood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --network.name coop_negoodprompt \
            --network.backbone.OOD_NUM ${OOD_NUM} \
            --network.backbone.text_prompt tip \
            --network.backbone.CSC False \
            --network.backbone.N_CTX 4 \
            --network.pretrained True \
            --network.checkpoint ./lapt_exp/debug_oodval_dedup_scratch/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.000/s10/best.ckpt \
            --evaluator.name ood_clip \
            --postprocessor.name oneoodprompt \
            --postprocessor.APS_mode False \
            --postprocessor.postprocessor_args.tau 1 \
            --postprocessor.postprocessor_args.in_score sum \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --dataset.interpolation bilinear \
            --dataset.train.batch_size 32 \
            --dataset.num_classes 11000 \
            --dataset.train.few_shot 0 \
            --seed ${seed} \
            --num_gpus 1 --num_workers 4 \
            --merge_option merge \
            --output_dir ./debug_test_lapt_release/ \
            --mark permute${random_permute}_group${group_num}_alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
    done
done

# 10K OOD, 100 group.
# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 196/196 [02:35<00:00,  1.26it/s]
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [01:30<00:00,  2.11it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 66.45, AUROC: 80.10 AUPR_IN: 80.53, AUPR_OUT: 78.80
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:26<00:00,  1.16s/it]
# Computing metrics on ninco dataset...
# FPR@95: 53.88, AUROC: 83.66 AUPR_IN: 97.66, AUPR_OUT: 46.11
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 60.16, AUROC: 81.88 AUPR_IN: 89.09, AUPR_OUT: 62.46
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:56<00:00,  1.41s/it]
# Computing metrics on inaturalist dataset...
# FPR@95: 1.10, AUROC: 99.60 AUPR_IN: 99.91, AUPR_OUT: 98.58
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Performing inference on sun dataset...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [05:27<00:00,  2.09s/it]
# Computing metrics on sun dataset...
# FPR@95: 20.59, AUROC: 95.35 AUPR_IN: 96.09, AUPR_OUT: 93.92
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Performing inference on places dataset...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [02:22<00:00,  1.10it/s]
# Computing metrics on places dataset...
# FPR@95: 35.38, AUROC: 92.84 AUPR_IN: 93.83, AUPR_OUT: 91.23
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Performing inference on dtd dataset...
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:21<00:00,  1.09it/s]
# Computing metrics on dtd dataset...
# FPR@95: 40.11, AUROC: 90.42 AUPR_IN: 98.72, AUPR_OUT: 57.79
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 24.29, AUROC: 94.55 AUPR_IN: 97.14, AUPR_OUT: 85.38
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────


### 1000 ood; 1 group
# Computing metrics on inaturalist dataset...
# FPR@95: 1.16, AUROC: 99.63 AUPR_IN: 99.92, AUPR_OUT: 98.49
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Computing metrics on sun dataset...
# FPR@95: 27.67, AUROC: 93.62 AUPR_IN: 95.86, AUPR_OUT: 89.18
# ACC: 67.91
# ──────────────────────────────────────────────────────────────────────
# Computing metrics on places dataset...
# FPR@95: 40.31, AUROC: 91.04 AUPR_IN: 94.14, AUPR_OUT: 85.69
# ──────────────────────────────────────────────────────────────────────
# Computing metrics on dtd dataset...
# FPR@95: 45.85, AUROC: 88.33 AUPR_IN: 98.35, AUPR_OUT: 49.28
# ACC: 67.91
 



# ##### test with pre-trained classifier weights
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# # mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm
# mix_strategy=wccm

# depth=5
# alpha=1
# beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
# total_gs_num=1000
# loss_components=textentropy
# gs_loss_weight=1-1-1-1
# OOD_NUM=1000
# shot_num=16
# weight_decay=0.
# optimizer=sgd
# # N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# # ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# # --evaluator.name fsood_clip \
# for seed in 0 
# do
#     for lr in 0.1 
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/test/test_ood.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.backbone.CSC False \
#             --network.backbone.N_CTX 4 \
#             --network.pretrained True \
#             --network.checkpoint ./debug_oodval_dedup_scratch/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.000/s10/best.ckpt \
#             --evaluator.name ood_clip \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.APS_mode False \
#             --postprocessor.postprocessor_args.tau 1 \
#             --postprocessor.postprocessor_args.in_score sum \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 32 \
#             --dataset.num_classes 2000 \
#             --dataset.train.few_shot 0 \
#             --seed ${seed} \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./debug_test/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
#     done
# done
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [01:45<00:00,  1.81it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 65.47, AUROC: 80.09 AUPR_IN: 79.13, AUPR_OUT: 80.28
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:17<00:00,  1.35it/s]
# Computing metrics on ninco dataset...
# FPR@95: 52.40, AUROC: 85.16 AUPR_IN: 97.65, AUPR_OUT: 49.32
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 58.94, AUROC: 82.63 AUPR_IN: 88.39, AUPR_OUT: 64.80
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:36<00:00,  1.10it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 1.17, AUROC: 99.63 AUPR_IN: 99.91, AUPR_OUT: 98.60
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:13<00:00,  1.54it/s]
# Computing metrics on textures dataset...
# FPR@95: 38.40, AUROC: 89.72 AUPR_IN: 98.61, AUPR_OUT: 51.81
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:35<00:00,  1.73it/s]
# Computing metrics on openimageo dataset...
# FPR@95: 35.00, AUROC: 93.45 AUPR_IN: 97.23, AUPR_OUT: 87.35
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 24.86, AUROC: 94.26 AUPR_IN: 98.58, AUPR_OUT: 79.25
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────



# ##### test with pre-trained classifier weights
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# # mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm
# mix_strategy=wccm

# depth=5
# alpha=1
# beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
# total_gs_num=1000
# loss_components=textentropy
# gs_loss_weight=1-1-1-1
# OOD_NUM=1000
# shot_num=16
# weight_decay=0.
# optimizer=sgd
# # N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# # ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# # --evaluator.name fsood_clip \
# for seed in 0 
# do
#     for lr in 0.1 
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/test/test_ood.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.backbone.CSC False \
#             --network.backbone.N_CTX 4 \
#             --network.pretrained True \
#             --network.checkpoint ./debug_oodval_dedup_scratch/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.000/s10/best.ckpt \
#             --evaluator.name fsood_clip \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.APS_mode False \
#             --postprocessor.postprocessor_args.tau 1 \
#             --postprocessor.postprocessor_args.in_score sum \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 32 \
#             --dataset.num_classes 2000 \
#             --dataset.train.few_shot 0 \
#             --seed ${seed} \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./debug_test/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
#     done
# done
# Start evaluation...
# Eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:38<00:00,  1.79it/s]

# Accuracy 67.86%
# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:37<00:00,  1.80it/s]
# Performing inference on imagenetv2 dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:23<00:00,  1.67it/s]
# Performing inference on imagenetc dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:22<00:00,  1.75it/s]
# Performing inference on imagenetr dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [01:06<00:00,  1.78it/s]
# ──────────────────────────────────────────────────────────────────────
# Computing accuracy on imagenetv2 dataset...
# CSID[imagenetv2] accuracy: 61.44%
# Computing accuracy on imagenetc dataset...
# CSID[imagenetc] accuracy: 40.39%
# Computing accuracy on imagenetr dataset...
# CSID[imagenetr] accuracy: 54.49%
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [01:45<00:00,  1.81it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 76.44, AUROC: 71.73 AUPR_IN: 82.92, AUPR_OUT: 56.90
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:15<00:00,  1.46it/s]
# Computing metrics on ninco dataset...
# FPR@95: 65.92, AUROC: 77.80 AUPR_IN: 98.17, AUPR_OUT: 28.42
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 71.18, AUROC: 74.77 AUPR_IN: 90.54, AUPR_OUT: 42.66
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:34<00:00,  1.16it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 2.06, AUROC: 99.40 AUPR_IN: 99.93, AUPR_OUT: 96.66
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:13<00:00,  1.56it/s]
# Computing metrics on textures dataset...
# FPR@95: 53.30, AUROC: 83.72 AUPR_IN: 98.87, AUPR_OUT: 26.04
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:35<00:00,  1.74it/s]
# Computing metrics on openimageo dataset...
# FPR@95: 49.83, AUROC: 89.99 AUPR_IN: 97.81, AUPR_OUT: 74.35
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 35.07, AUROC: 91.04 AUPR_IN: 98.87, AUPR_OUT: 65.68
# ACC: 60.07
# ──────────────────────────────────────────────────────────────────────


    # --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood11k_syn2_filtered_acc_pred.txt \
            # --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood11k_syn2_filtered.txt \
                # --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood11k_syn32.txt \
        # --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood11k_syn32_filtered_acc_pred.txt \
    # 

# ############################################################  Final, zero-shot performance.
# tau=1.0
# beta=1.0
# ######################### clip + negood +  Neglabel, OOD.
# in_score=sum
# prompt=simple
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
#     #    --evaluator.name ood_clip \
# for prompt in vanilla good bad small large simple nice
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./various_prompt/ \
#         --mark ${in_score}_beta${beta}_${prompt}
#     done
# done
# ############################################ neglabel, FSOOD
# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# in_score=sum
# prompt=simple
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
#     #    --evaluator.name ood_clip \
# for prompt in nice
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --evaluator.name fsood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./various_prompt_fsood/ \
#         --mark ${in_score}_beta${beta}_${prompt}
#     done
# done
# Start evaluation...
# Eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.54it/s]

# Accuracy 66.77%
# ──────────────────────────────────────────────────────────────────────
# Performing inference on imagenet dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.49it/s]
# Performing inference on imagenetv2 dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:17<00:00,  2.25it/s]
# Performing inference on imagenetc dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:10<00:00,  3.64it/s]
# Performing inference on imagenetr dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:47<00:00,  2.47it/s]
# ──────────────────────────────────────────────────────────────────────
# Computing accuracy on imagenetv2 dataset...
# CSID[imagenetv2] accuracy: 60.90%
# Computing accuracy on imagenetc dataset...
# CSID[imagenetc] accuracy: 39.68%
# Computing accuracy on imagenetr dataset...
# CSID[imagenetr] accuracy: 56.67%
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:47<00:00,  4.06it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 83.91, AUROC: 68.56 AUPR_IN: 79.66, AUPR_OUT: 54.32
# ACC: 60.11
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.65it/s]
# Computing metrics on ninco dataset...
# FPR@95: 68.59, AUROC: 76.99 AUPR_IN: 98.05, AUPR_OUT: 29.16
# ACC: 60.11
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 76.25, AUROC: 72.77 AUPR_IN: 88.86, AUPR_OUT: 41.74
# ACC: 60.11
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.40it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 1.25, AUROC: 99.56 AUPR_IN: 99.95, AUPR_OUT: 97.46
# ACC: 60.11
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:09<00:00,  2.11it/s]
# Computing metrics on textures dataset...
# FPR@95: 53.59, AUROC: 85.97 AUPR_IN: 99.02, AUPR_OUT: 27.97
# ACC: 60.11
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:22<00:00,  2.70it/s]
# Computing metrics on openimageo dataset...
# FPR@95: 45.04, AUROC: 90.54 AUPR_IN: 98.02, AUPR_OUT: 73.66
# ACC: 60.11
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 33.30, AUROC: 92.02 AUPR_IN: 98.99, AUPR_OUT: 66.36
# ACC: 60.11



# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# prompt=nice
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in sum
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done

# # Start evaluation...
# # Eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.53it/s]

# # Accuracy 66.77%
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on imagenet dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.50it/s]
# # ──────────────────────────────────────────────────────────────────────
# # Processing nearood...
# # Performing inference on ssb_hard dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:49<00:00,  3.91it/s]
# # Computing metrics on ssb_hard dataset...
# # FPR@95: 77.29, AUROC: 73.14 AUPR_IN: 71.28, AUPR_OUT: 73.32
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on ninco dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.69it/s]
# # Computing metrics on ninco dataset...
# # FPR@95: 59.39, AUROC: 80.76 AUPR_IN: 96.88, AUPR_OUT: 42.25
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 68.34, AUROC: 76.95 AUPR_IN: 84.08, AUPR_OUT: 57.79
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing farood...
# # Performing inference on inaturalist dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.38it/s]
# # Computing metrics on inaturalist dataset...
# # FPR@95: 1.46, AUROC: 99.57 AUPR_IN: 99.90, AUPR_OUT: 98.36
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on textures dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:10<00:00,  2.06it/s]
# # Computing metrics on textures dataset...
# # FPR@95: 44.34, AUROC: 88.17 AUPR_IN: 98.38, AUPR_OUT: 44.34
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on openimageo dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:22<00:00,  2.70it/s]
# # Computing metrics on openimageo dataset...
# # FPR@95: 36.59, AUROC: 92.04 AUPR_IN: 96.81, AUPR_OUT: 83.65
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 27.46, AUROC: 93.26 AUPR_IN: 98.36, AUPR_OUT: 75.45
# # ACC: 66.77
# # ──────────────────────────────────────────────────────────────────────

# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  Neglabel.
# prompt=nice
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in sum
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 2000 \
#         --evaluator.name fsood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done

# # Eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.51it/s]

# # Accuracy 66.77%
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on imagenet dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.52it/s]
# # Performing inference on imagenetv2 dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:18<00:00,  2.20it/s]
# # Performing inference on imagenetc dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:11<00:00,  3.62it/s]
# # Performing inference on imagenetr dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:48<00:00,  2.42it/s]
# # ──────────────────────────────────────────────────────────────────────
# # Computing accuracy on imagenetv2 dataset...
# # CSID[imagenetv2] accuracy: 60.90%
# # Computing accuracy on imagenetc dataset...
# # CSID[imagenetc] accuracy: 39.68%
# # Computing accuracy on imagenetr dataset...
# # CSID[imagenetr] accuracy: 56.67%
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing nearood...
# # Performing inference on ssb_hard dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:47<00:00,  4.05it/s]
# # Computing metrics on ssb_hard dataset...
# # FPR@95: 83.91, AUROC: 68.56 AUPR_IN: 79.66, AUPR_OUT: 54.32
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on ninco dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.66it/s]
# # Computing metrics on ninco dataset...
# # FPR@95: 68.59, AUROC: 76.99 AUPR_IN: 98.05, AUPR_OUT: 29.16
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 76.25, AUROC: 72.77 AUPR_IN: 88.86, AUPR_OUT: 41.74
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing farood...
# # Performing inference on inaturalist dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.41it/s]
# # Computing metrics on inaturalist dataset...
# # FPR@95: 1.25, AUROC: 99.56 AUPR_IN: 99.95, AUPR_OUT: 97.46
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on textures dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:09<00:00,  2.13it/s]
# # Computing metrics on textures dataset...
# # FPR@95: 53.59, AUROC: 85.97 AUPR_IN: 99.02, AUPR_OUT: 27.97
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on openimageo dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:22<00:00,  2.74it/s]
# # Computing metrics on openimageo dataset...
# # FPR@95: 45.04, AUROC: 90.54 AUPR_IN: 98.02, AUPR_OUT: 73.66
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 33.30, AUROC: 92.02 AUPR_IN: 98.99, AUPR_OUT: 66.36
# # ACC: 60.11
# # ──────────────────────────────────────────────────────────────────────



# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# prompt=simple
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in sum
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 2 \
#         --dataset.num_classes 1000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name mcm \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done

# # Accuracy 66.28%
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on imagenet dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.52it/s]
# # ──────────────────────────────────────────────────────────────────────
# # Processing nearood...
# # Performing inference on ssb_hard dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:49<00:00,  3.89it/s]
# # Computing metrics on ssb_hard dataset...
# # FPR@95: 86.25, AUROC: 63.01 AUPR_IN: 60.88, AUPR_OUT: 63.41
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on ninco dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:14<00:00,  1.62it/s]
# # Computing metrics on ninco dataset...
# # FPR@95: 71.80, AUROC: 74.06 AUPR_IN: 95.45, AUPR_OUT: 27.25
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 79.02, AUROC: 68.54 AUPR_IN: 78.16, AUPR_OUT: 45.33
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing farood...
# # Performing inference on inaturalist dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.38it/s]
# # Computing metrics on inaturalist dataset...
# # FPR@95: 56.03, AUROC: 86.41 AUPR_IN: 96.24, AUPR_OUT: 62.56
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on textures dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:09<00:00,  2.12it/s]
# # Computing metrics on textures dataset...
# # FPR@95: 67.48, AUROC: 81.83 AUPR_IN: 97.03, AUPR_OUT: 40.86
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on openimageo dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:23<00:00,  2.66it/s]
# # Computing metrics on openimageo dataset...
# # FPR@95: 56.82, AUROC: 86.06 AUPR_IN: 94.04, AUPR_OUT: 71.05
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 60.11, AUROC: 84.77 AUPR_IN: 95.77, AUPR_OUT: 58.16
# # ACC: 66.28
# # ──────────────────────────────────────────────────────────────────────

# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# prompt=simple
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in sum
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 2 \
#         --dataset.num_classes 1000 \
#         --evaluator.name fsood_clip \
#         --network.name fixedclip \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name mcm \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_test/ \
#         --mark ${in_score}_beta${beta}
#     done
# done

# # Start evaluation...
# # Eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.49it/s]

# # Accuracy 66.28%
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on imagenet dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.50it/s]
# # Performing inference on imagenetv2 dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:18<00:00,  2.17it/s]
# # Performing inference on imagenetc dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:11<00:00,  3.54it/s]
# # Performing inference on imagenetr dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:48<00:00,  2.43it/s]
# # ──────────────────────────────────────────────────────────────────────
# # Computing accuracy on imagenetv2 dataset...
# # CSID[imagenetv2] accuracy: 60.46%
# # Computing accuracy on imagenetc dataset...
# # CSID[imagenetc] accuracy: 39.71%
# # Computing accuracy on imagenetr dataset...
# # CSID[imagenetr] accuracy: 49.87%
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing nearood...
# # Performing inference on ssb_hard dataset...
# # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:47<00:00,  4.03it/s]
# # Computing metrics on ssb_hard dataset...
# # FPR@95: 90.91, AUROC: 53.44 AUPR_IN: 70.27, AUPR_OUT: 34.68
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on ninco dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:14<00:00,  1.63it/s]
# # Computing metrics on ninco dataset...
# # FPR@95: 79.82, AUROC: 64.49 AUPR_IN: 96.78, AUPR_OUT: 8.36
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 85.37, AUROC: 58.97 AUPR_IN: 83.53, AUPR_OUT: 21.52
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────
# # ──────────────────────────────────────────────────────────────────────
# # Processing farood...
# # Performing inference on inaturalist dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.38it/s]
# # Computing metrics on inaturalist dataset...
# # FPR@95: 66.35, AUROC: 78.96 AUPR_IN: 97.08, AUPR_OUT: 25.29
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on textures dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:10<00:00,  1.98it/s]
# # Computing metrics on textures dataset...
# # FPR@95: 76.22, AUROC: 73.76 AUPR_IN: 97.87, AUPR_OUT: 12.04
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────
# # Performing inference on openimageo dataset...
# # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:23<00:00,  2.67it/s]
# # Computing metrics on openimageo dataset...
# # FPR@95: 67.05, AUROC: 78.59 AUPR_IN: 95.36, AUPR_OUT: 34.81
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────
# # Computing mean metrics...
# # FPR@95: 69.87, AUROC: 77.11 AUPR_IN: 96.77, AUPR_OUT: 24.05
# # ACC: 57.69
# # ──────────────────────────────────────────────────────────────────────




# ############################################################ 
# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# prompt=tip
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in guiscore_text_norm_merge guiscore_text_add_merge guiscore_text_cosonly_merge guiscore_text_cosonly_mod guiscore_text_cosonly_mod_merge
# do
#     for beta in 1 2 3 5 10 40 100 200 500 1000 3000
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood11k_syn2_filtered_acc_pred.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 11000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug/ \
#         --mark ${in_score}_beta${beta}
#     done
# done








##################################### baseline.
# for tau in 5
# do
#     ######################### clip + MCM.
#     python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/fixed_clip.yml \
#     configs/pipelines/test/test_fsood.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --network.backbone.text_prompt simple \
#     --network.pretrained False \
#     --postprocessor.postprocessor_args.tau ${tau}  \
#     --postprocessor.postprocessor_args.in_score sum  \
#     --dataset.train.batch_size 128 \
#     --dataset.train.few_shot 16 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --output_dir ./test_fixedclip/ \
#     --mark tau${tau}
# done 

# mix_strategy=manimix_wccm
# alpha=1
# beta=1
# for tau in 1
# do
#     for beta in 1
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         configs/networks/maple.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/she.yml \
#         --network.name maple_oneoodprompt \
#         --network.backbone.text_prompt tip \
#         --network.pretrained True \
#         --network.checkpoint ./exp_oodnum/imagenet_maple_oneoodprompt_oodmixup_sgd_e60_lr0.1_alpha1_beta1_manimix_wccm_pdepth5_oodnum1_1-1-1-0.1/s0/best.ckpt \
#         --postprocessor.name knnoneoodprompt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 16 \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug/ \
#         --mark alpha${alpha}_beta${beta}_${mix_strategy}_test_debug_tau${tau}

#         # --network.checkpoint ./exp_feat_mix/imagenet_coop_oneoodprompt_oodmixup_sgd_e100_lr0.1_alpha${alpha}_beta${beta}_${mix_strategy}/s0/best.ckpt \
#         # knnoneoodprompt   sheoneoodprompt
#         ########## prior best.
#         #         --network.checkpoint ./exp_rerun_coop/imagenet_coop_oneoodprompt_oodmixup_sgd_e100_lr0.1_alpha1_beta1_manimix_wccm_classify_lossw1-1-1-1/s0/best.ckpt \

#         # python main.py \
#         #     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#         #     configs/networks/coop.yml \
#         #     configs/pipelines/train/train_coop.yml \
#         #     configs/preprocessors/base_preprocessor.yml \
#         #     configs/postprocessors/mcm.yml \
#         #     --network.name coop_oneoodprompt \
#         #     --network.backbone.text_prompt tip \
#         #     --network.pretrained False \
#         #     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#         #     --trainer.trainer_args.alpha ${alpha} \
#         #     --trainer.trainer_args.beta ${beta} \
#         #     --trainer.trainer_args.mix_strategy ${mix_strategy} \
#         #     --trainer.name oodmixup \
#         #     --postprocessor.name oneoodprompt \
#         #     --postprocessor.postprocessor_args.tau 1  \
#         #     --postprocessor.postprocessor_args.in_score sum  \
#         #     --dataset.train.batch_size 128 \
#         #     --dataset.train.few_shot 16 \
#         #     --optimizer.num_epochs 100 \
#         #     --optimizer.lr 0.1 \
#         #     --num_gpus 1 --num_workers 6 \
#         #     --merge_option merge \
#         #     --output_dir ./exp_beta_value/ \
#         #     --mark alpha${alpha}_beta${beta}_${mix_strategy} \
#         #     --seed 0
#     done
# done