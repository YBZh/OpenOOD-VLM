# vim ./debug_oodval_dedup_scratch_nctx2/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.00001/s10/log.txt

# Starting automatic parameter search...
# Hyperparam:[0.7], auroc:0.8578397006293587
# Hyperparam:[1], auroc:0.8595480693995576
# Hyperparam:[1.5], auroc:0.8547100187106651
# Final hyperparam: 1
# Performing inference on imagenet dataset...
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# Computing metrics on ssb_hard dataset...
# FPR@95: 62.97, AUROC: 80.71 AUPR_IN: 79.97, AUPR_OUT: 80.13
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# Computing metrics on ninco dataset...
# FPR@95: 50.08, AUROC: 85.68 AUPR_IN: 97.76, AUPR_OUT: 50.46
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 56.53, AUROC: 83.20 AUPR_IN: 88.87, AUPR_OUT: 65.30
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# Computing metrics on inaturalist dataset...
# FPR@95: 0.66, AUROC: 99.76 AUPR_IN: 99.94, AUPR_OUT: 99.08
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# Computing metrics on textures dataset...
# FPR@95: 49.60, AUROC: 85.65 AUPR_IN: 97.98, AUPR_OUT: 40.34
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# Computing metrics on openimageo dataset...
# FPR@95: 29.83, AUROC: 94.05 AUPR_IN: 97.65, AUPR_OUT: 87.32
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 26.70, AUROC: 93.15 AUPR_IN: 98.52, AUPR_OUT: 75.58
# ACC: 67.68
# ──────────────────────────────────────────────────────────────────────
# Completed!



# vim ./debug_oodval_dedup_scratch/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.000/s10/log.txt

# Complete Evaluation, accuracy 67.86
# Performing inference on imagenet dataset...
# ──────────────────────────────────────────────────────────────────────
# Processing nearood...
# Performing inference on ssb_hard dataset...
# Computing metrics on ssb_hard dataset...
# FPR@95: 65.47, AUROC: 80.09 AUPR_IN: 79.13, AUPR_OUT: 80.28
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
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
# Computing metrics on inaturalist dataset...
# FPR@95: 1.17, AUROC: 99.63 AUPR_IN: 99.91, AUPR_OUT: 98.60
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Performing inference on textures dataset...
# Computing metrics on textures dataset...
# FPR@95: 38.40, AUROC: 89.72 AUPR_IN: 98.61, AUPR_OUT: 51.81
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Performing inference on openimageo dataset...
# Computing metrics on openimageo dataset...
# FPR@95: 35.00, AUROC: 93.45 AUPR_IN: 97.23, AUPR_OUT: 87.35
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 24.86, AUROC: 94.26 AUPR_IN: 98.58, AUPR_OUT: 79.25
# ACC: 67.86
# ──────────────────────────────────────────────────────────────────────
# Completed!


             
############################################################################ we can get the paper results with the code here. 
#!/bin/bash
# sh scripts/ood/coop/imagenet_train_coop.sh

################################### ablate ood prompt number and ood entropy loss.
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
loss_components=multice
gs_loss_weight=1-1-1-1
OOD_NUM=1000
shot_num=16
weight_decay=0.00001
optimizer=sgd
N_CTX=2
# N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
############ 
for seed in 0 10
do
    for lr in 0.01
    do
        for weight_decay 0.00001 
        do
            python main.py \
            --config configs/datasets/imagenet/imagenet_train_fsood.yml \
            configs/networks/coop.yml \
            configs/pipelines/train/train_coop.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --network.name coop_negoodprompt \
            --network.backbone.OOD_NUM ${OOD_NUM} \
            --network.backbone.text_prompt tip \
            --network.pretrained False \
            --network.backbone.CSC False \
            --network.backbone.N_CTX ${N_CTX} \
            --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
            --trainer.trainer_args.alpha ${alpha} \
            --trainer.trainer_args.beta ${beta} \
            --trainer.trainer_args.mix_strategy ${mix_strategy} \
            --trainer.trainer_args.total_gs_num ${total_gs_num} \
            --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
            --trainer.trainer_args.loss_components ${loss_components} \
            --trainer.trainer_args.gs_flag False \
            --trainer.trainer_args.queue_capacity 500 \
            --trainer.trainer_args.pre_queue True \
            --trainer.trainer_args.iter_recompuration 10 \
            --trainer.trainer_args.soft_split True \
            --evaluator.name ood_clip \
            --trainer.name oodmixup_idood \
            --postprocessor.name oneoodprompt \
            --postprocessor.APS_mode True \
            --postprocessor.postprocessor_args.tau 1 \
            --postprocessor.postprocessor_args.in_score oodscore \
            --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2_dedup.txt \
            --dataset.interpolation bilinear \
            --dataset.train.batch_size 32 \
            --dataset.num_classes 2000 \
            --dataset.train.few_shot 0 \
            --seed ${seed} \
            --optimizer.name ${optimizer} \
            --optimizer.num_epochs 10 \
            --optimizer.lr ${lr} \
            --optimizer.weight_decay ${weight_decay} \
            --num_gpus 1 --num_workers 4 \
            --merge_option merge \
            --output_dir ./debug_oodval_dedup_scratch_nctx${N_CTX}/ \
            --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
        done
    done
done