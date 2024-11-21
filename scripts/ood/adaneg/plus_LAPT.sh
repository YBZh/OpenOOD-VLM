##### apply AdaNeg (NIPS2024) to LAPT(ECCV2024)
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
group_num=5
random_permute=True
seed=0
gap=0.5
for group_num in 5 
do
    for lr in 0.1 
    do
        python main.py \
            --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
            configs/networks/coop.yml \
            configs/pipelines/test/test_ood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --dataset.num_classes 11000 \
            --ood_dataset.num_classes 11000 \
            --dataset.train.few_shot 0 \
            --network.name coop_negoodprompt \
            --network.backbone.OOD_NUM ${OOD_NUM} \
            --network.backbone.text_prompt simple \
            --network.backbone.CSC False \
            --network.backbone.N_CTX 4 \
            --network.pretrained True \
            --network.checkpoint ./lapt_exp/debug_oodval_dedup_scratch/imagenet_coop_negoodprompt_oodmixup_idood_sgd_e10_lr0.01_alpha1_beta0_wccm_ood1000_w1-1-1-1_multice_sgdlr0.01_wd0.000/s10/best.ckpt \
            --evaluator.name ood_clip_tta \
            --postprocessor.name ttapromptnoadagap \
            --postprocessor.APS_mode False \
            --postprocessor.postprocessor_args.tau 1 \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --postprocessor.postprocessor_args.thres 0.5  \
            --postprocessor.postprocessor_args.gap ${gap}  \
            --postprocessor.postprocessor_args.samada False  \
            --postprocessor.postprocessor_args.memleng 10  \
            --postprocessor.postprocessor_args.lambdaval 0.1  \
            --postprocessor.postprocessor_args.beta 5.5  \
            --postprocessor.postprocessor_args.in_score combine  \
            --dataset.interpolation bilinear \
            --dataset.train.batch_size 32 \
            --dataset.num_classes 11000 \
            --dataset.train.few_shot 0 \
            --seed ${seed} \
            --num_gpus 1 --num_workers 4 \
            --merge_option merge \
            --output_dir ./debug_test_lapt_release/ \
            --mark gap${gap}_permute${random_permute}_group${group_num}_alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}_adaneg
    done
done

# Processing nearood...
# Performing inference on ssb_hard dataset...
# Generating combined dataset with ID and OOD dataset of ssb_hard, total size 99000
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 387/387 [04:22<00:00,  1.47it/s]
# Computing metrics on ssb_hard dataset...
# FPR@95: 62.50, AUROC: 82.68 AUPR_IN: 82.80, AUPR_OUT: 81.56
# ACC: 67.41
# ──────────────────────────────────────────────────────────────────────
# Performing inference on ninco dataset...
# Generating combined dataset with ID and OOD dataset of ninco, total size 55879
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 219/219 [03:10<00:00,  1.15it/s]
# Computing metrics on ninco dataset...
# FPR@95: 53.43, AUROC: 85.36 AUPR_IN: 97.89, AUPR_OUT: 48.06
# ACC: 67.47
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 57.97, AUROC: 84.02 AUPR_IN: 90.35, AUPR_OUT: 64.81
# ACC: 67.44
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# Processing farood...
# Performing inference on inaturalist dataset...
# Generating combined dataset with ID and OOD dataset of inaturalist, total size 60000
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [03:43<00:00,  1.05it/s]
# Computing metrics on inaturalist dataset...
# FPR@95: 0.58, AUROC: 99.76 AUPR_IN: 99.95, AUPR_OUT: 99.21
# ACC: 67.23
# ──────────────────────────────────────────────────────────────────────
# Performing inference on sun dataset...
# Generating combined dataset with ID and OOD dataset of sun, total size 60000
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [04:14<00:00,  1.08s/it]
# Computing metrics on sun dataset...
# FPR@95: 9.98, AUROC: 97.67 AUPR_IN: 99.42, AUPR_OUT: 91.67
# ACC: 67.32
# ──────────────────────────────────────────────────────────────────────
# Performing inference on places dataset...
# Generating combined dataset with ID and OOD dataset of places, total size 60000
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 235/235 [03:26<00:00,  1.14it/s]
# Computing metrics on places dataset...
# FPR@95: 30.47, AUROC: 94.93 AUPR_IN: 98.72, AUPR_OUT: 84.54
# ACC: 67.34
# ──────────────────────────────────────────────────────────────────────
# Performing inference on dtd dataset...
# Generating combined dataset with ID and OOD dataset of dtd, total size 55640
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 218/218 [03:06<00:00,  1.17it/s]
# Computing metrics on dtd dataset...
# FPR@95: 24.25, AUROC: 95.66 AUPR_IN: 99.39, AUPR_OUT: 80.61
# ACC: 67.48
# ──────────────────────────────────────────────────────────────────────
# Computing mean metrics...
# FPR@95: 16.32, AUROC: 97.01 AUPR_IN: 99.37, AUPR_OUT: 89.01
# ACC: 67.34