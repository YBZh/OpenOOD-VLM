
# ############################################################ 
tau=1.0
beta=1.0
######################### clip + ood + NegLabel
prompt=nice
random_permute=True
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
in_score=sum
group_num=100
ood_number=10000
for beta in 1 2 3 5 10 20 50 100
do
    for in_score in div_score_prior
    do
        python main.py \
        --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
        configs/networks/fixed_clip.yml \
        configs/pipelines/test/test_fsood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/mcm.yml \
        --dataset.train.batch_size 128 \
        --dataset.train.few_shot 0 \
        --dataset.num_classes $((ood_number + 1000)) \
        --ood_dataset.num_classes $((ood_number + 1000)) \
        --evaluator.name ood_clip_tta \
        --network.name fixedclip_negoodprompt \
        --network.backbone.ood_number  ${ood_number} \
        --network.backbone.text_prompt ${prompt} \
        --network.backbone.text_center True \
        --network.pretrained False \
        --postprocessor.APS_mode False \
        --postprocessor.name labelrelationship \
        --postprocessor.postprocessor_args.group_num ${group_num}  \
        --postprocessor.postprocessor_args.random_permute ${random_permute}  \
        --postprocessor.postprocessor_args.tau ${tau}  \
        --postprocessor.postprocessor_args.beta ${beta}  \
        --postprocessor.postprocessor_args.in_score ${in_score}  \
        --num_gpus 1 --num_workers 6 \
        --merge_option merge \
        --output_dir ./dug_catewise_thres/ \
        --mark ${in_score}_beta${beta}_neg10k_ood_number_${ood_number}_random_${random_permute}
    done
done


for beta in 1 2 3 5 10 20 50 100
do
    for in_score in div_count
    do
        python main.py \
        --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
        configs/networks/fixed_clip.yml \
        configs/pipelines/test/test_fsood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/mcm.yml \
        --dataset.train.batch_size 128 \
        --dataset.train.few_shot 0 \
        --dataset.num_classes $((ood_number + 1000)) \
        --ood_dataset.num_classes $((ood_number + 1000)) \
        --evaluator.name ood_clip_tta \
        --network.name fixedclip_negoodprompt \
        --network.backbone.ood_number  ${ood_number} \
        --network.backbone.text_prompt ${prompt} \
        --network.backbone.text_center True \
        --network.pretrained False \
        --postprocessor.APS_mode False \
        --postprocessor.name labelrelationship \
        --postprocessor.postprocessor_args.group_num ${group_num}  \
        --postprocessor.postprocessor_args.random_permute ${random_permute}  \
        --postprocessor.postprocessor_args.tau ${tau}  \
        --postprocessor.postprocessor_args.beta ${beta}  \
        --postprocessor.postprocessor_args.in_score ${in_score}  \
        --num_gpus 1 --num_workers 6 \
        --merge_option merge \
        --output_dir ./dug_catewise_thres/ \
        --mark ${in_score}_beta${beta}_neg10k_ood_number_${ood_number}_random_${random_permute}_count_latter
    done
done


# Processing nearood...
# Performing inference on ssb_hard dataset...
# Generating combined dataset with ID and OOD dataset of ssb_hard, total size 99000
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 387/387 [03:34<00:00,  1.81it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 79.86, AUROC: 69.19 AUPR_IN: 70.00, AUPR_OUT: 67.18
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# Generating combined dataset with ID and OOD dataset of ninco, total size 55879
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 219/219 [02:38<00:00,  1.39it/s]
# Computing metrics on ninco dataset...
# FPR@95: 63.91, AUROC: 78.20 AUPR_IN: 96.68, AUPR_OUT: 37.74
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 71.89, AUROC: 73.70 AUPR_IN: 83.34, AUPR_OUT: 52.46
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# Generating combined dataset with ID and OOD dataset of inaturalist, total size 60000
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [03:03<00:00,  1.28it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 2.04, AUROC: 99.43 AUPR_IN: 99.88, AUPR_OUT: 97.47
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# Performing inference on sun dataset...
# Generating combined dataset with ID and OOD dataset of sun, total size 60000
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [03:33<00:00,  1.10it/s]
# Computing metrics on sun dataset...
# FPR@95: 34.79, AUROC: 91.46 AUPR_IN: 98.08, AUPR_OUT: 65.65
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# Performing inference on places dataset...
# Generating combined dataset with ID and OOD dataset of places, total size 60000
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [02:56<00:00,  1.33it/s]
# Computing metrics on places dataset...
# FPR@95: 45.25, AUROC: 88.16 AUPR_IN: 97.20, AUPR_OUT: 59.16
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# Performing inference on dtd dataset...
# Generating combined dataset with ID and OOD dataset of dtd, total size 55640
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 218/218 [02:36<00:00,  1.39it/s]
# Computing metrics on dtd dataset...
# FPR@95: 38.02, AUROC: 90.08 AUPR_IN: 98.70, AUPR_OUT: 51.39
# ACC: 66.85
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 30.03, AUROC: 92.28 AUPR_IN: 98.46, AUPR_OUT: 68.42
# ACC: 66.85



######################################## OOD datasets under OpenOOD setting.
# tau=1.0
# beta=1.0
# ######################### clip + ood + NegLabel
# prompt=nice
# random_permute=True
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# # --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
# #         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
# for in_score in sum
# do
#     for group_num in 100
#     do
#         python main.py \
#         --config configs/datasets/imagenet/imagenet_train_ood.yml \
#         configs/networks/fixed_clip.yml \
#         configs/pipelines/test/test_fsood.yml \
#         configs/preprocessors/base_preprocessor.yml \
#         configs/postprocessors/mcm.yml \
#         --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#         --dataset.train.batch_size 128 \
#         --dataset.train.few_shot 0 \
#         --dataset.num_classes 11000 \
#         --evaluator.name ood_clip \
#         --network.name fixedclip_negoodprompt \
#         --network.backbone.ood_number 10000 \
#         --network.backbone.text_prompt ${prompt} \
#         --network.backbone.text_center True \
#         --network.pretrained False \
#         --postprocessor.APS_mode False \
#         --postprocessor.name oneoodpromptdevelop \
#         --postprocessor.postprocessor_args.group_num ${group_num}  \
#         --postprocessor.postprocessor_args.random_permute ${random_permute}  \
#         --postprocessor.postprocessor_args.tau ${tau}  \
#         --postprocessor.postprocessor_args.beta ${beta}  \
#         --postprocessor.postprocessor_args.in_score ${in_score}  \
#         --num_gpus 1 --num_workers 6 \
#         --merge_option merge \
#         --output_dir ./debug_neglabel_official/ \
#         --mark ${in_score}_beta${beta}_neg10k_group_num_${group_num}_random_${random_permute}_official
#     done
# done


### network.backbone.ood_number 10000 Vs network.backbone.ood_number 1000;   10000 neglabels: better far ood , worse near ood 

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
# ######################### clip + fsood +  Neglabel.
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
#         --mark ${in_score}_beta${beta}_fsood
#     done
# done

# # # Eval: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:10<00:00,  2.51it/s]

# # # Accuracy 66.77%
# # # ──────────────────────────────────────────────────────────────────────
# # # Performing inference on imagenet dataset...
# # # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [01:09<00:00,  2.52it/s]
# # # Performing inference on imagenetv2 dataset...
# # # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:18<00:00,  2.20it/s]
# # # Performing inference on imagenetc dataset...
# # # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:11<00:00,  3.62it/s]
# # # Performing inference on imagenetr dataset...
# # # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 118/118 [00:48<00:00,  2.42it/s]
# # # ──────────────────────────────────────────────────────────────────────
# # # Computing accuracy on imagenetv2 dataset...
# # # CSID[imagenetv2] accuracy: 60.90%
# # # Computing accuracy on imagenetc dataset...
# # # CSID[imagenetc] accuracy: 39.68%
# # # Computing accuracy on imagenetr dataset...
# # # CSID[imagenetr] accuracy: 56.67%
# # # ──────────────────────────────────────────────────────────────────────
# # # ──────────────────────────────────────────────────────────────────────
# # # Processing nearood...
# # # Performing inference on ssb_hard dataset...
# # # 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 192/192 [00:47<00:00,  4.05it/s]
# # # Computing metrics on ssb_hard dataset...
# # # FPR@95: 83.91, AUROC: 68.56 AUPR_IN: 79.66, AUPR_OUT: 54.32
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────
# # # Performing inference on ninco dataset...
# # # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 23/23 [00:13<00:00,  1.66it/s]
# # # Computing metrics on ninco dataset...
# # # FPR@95: 68.59, AUROC: 76.99 AUPR_IN: 98.05, AUPR_OUT: 29.16
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────
# # # Computing mean metrics...
# # # FPR@95: 76.25, AUROC: 72.77 AUPR_IN: 88.86, AUPR_OUT: 41.74
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────
# # # ──────────────────────────────────────────────────────────────────────
# # # Processing farood...
# # # Performing inference on inaturalist dataset...
# # # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:28<00:00,  1.41it/s]
# # # Computing metrics on inaturalist dataset...
# # # FPR@95: 1.25, AUROC: 99.56 AUPR_IN: 99.95, AUPR_OUT: 97.46
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────
# # # Performing inference on textures dataset...
# # # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21/21 [00:09<00:00,  2.13it/s]
# # # Computing metrics on textures dataset...
# # # FPR@95: 53.59, AUROC: 85.97 AUPR_IN: 99.02, AUPR_OUT: 27.97
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────
# # # Performing inference on openimageo dataset...
# # # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62/62 [00:22<00:00,  2.74it/s]
# # # Computing metrics on openimageo dataset...
# # # FPR@95: 45.04, AUROC: 90.54 AUPR_IN: 98.02, AUPR_OUT: 73.66
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────
# # # Computing mean metrics...
# # # FPR@95: 33.30, AUROC: 92.02 AUPR_IN: 98.99, AUPR_OUT: 66.36
# # # ACC: 60.11
# # # ──────────────────────────────────────────────────────────────────────