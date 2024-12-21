
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
# N_CTX  should be identical to CTX_INIT !!!
# ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# --network.checkpoint ./lapt_exp/debug_oodval_dedup_scratch/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.000/s10/best.ckpt \
# --evaluator.name fsood_clip \
# to conduct experiments with datasets of OpenOOD, please change: imagenet_traditional_four_ood.yml to other configs.
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