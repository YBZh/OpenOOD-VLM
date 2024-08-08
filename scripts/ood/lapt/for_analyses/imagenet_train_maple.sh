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
############ learning rate 比较大的时候需要比较多的
for seed in 0 10
do
    for lr in 0.01 0.001 
    do
        for weight_decay in 0.01 0.1 0 0.00001 0.0001 0.001 
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
            --network.backbone.prompttype unified \
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
            --output_dir ./debug_oodval_dedup_scratch_nctx${N_CTX}_unified/ \
            --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_${loss_components}_${optimizer}lr${lr}_wd${weight_decay}
        done
    done
done

# ################################### ablate ood prompt number and ood entropy loss.
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# # mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm
# mix_strategy=vanilla

# depth=5
# alpha=1
# beta=0 ## beta==0 --> vanilla mixup; beta > 0, mixed data are given partial ood labels. 
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-0.1-1-1
# OOD_NUM=1000
# shot_num=16

# # N_CTX  要和 CTX_INIT 对应数量，否则会出问题!!!
# # ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
# for seed in 0 
# do
#     for lr in 0.00001 0.000001 0.0000001 
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/coop.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name coop_negoodprompt \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.backbone.CSC False \
#             --network.backbone.N_CTX 4 \
#             --network.backbone.CTX_INIT 'a photo of a' \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#             --trainer.trainer_args.soft_split True \
#             --evaluator.name ood_clip \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1 \
#             --postprocessor.postprocessor_args.in_score sum \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/train_imagenet_idood_last1k_syn2.txt \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.num_classes 2000 \
#             --dataset.train.few_shot 16 \
#             --seed ${seed} \
#             --optimizer.num_epochs 50 \
#             --optimizer.lr ${lr} \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./debug_synneg_pt/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_ood${OOD_NUM}_w${gs_loss_weight}_16shot_lr${lr}
#     done
# done



# tau=1.0
# beta=1.0
# ######################### clip + negood +  MCM.
# prompt=tip
# #  30 50 100 200 1000 2000
# # guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# for in_score in guiscore_img_cosonly_mod
# do
#     for beta in 2 5 10 40 100 200 500
#     do
#        for tau in 0.3 1 3 5.5 7.5 10 50
#        do
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
#         --output_dir ./debug_image_tau/ \
#         --mark ${in_score}_tau${beta}_tau${tau}
#         done
#     done
# done



# #################################### ablate ood prompt number and ood entropy loss.
# # mix_strategy=mixup
# # mix_strategy=cutmix
# # mix_strategy=manimix
# # mix_strategy=geomix
# # mix_strategy=manimix_wccm
# mix_strategy=geomix_wccm
# # mix_strategy=mixup_manimix_wccm
# # mix_strategy=mixup_geomix_wccm

# depth=5
# alpha=1
# beta=1
# total_gs_num=1000
# loss_components=classify
# gs_loss_weight=1-1-1-1
# N_CTX=16
# OOD_NUM=1
# for seed in 2 1 0
# do
#     for beta in 0 1 2
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/maple.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet.txt \
#             --network.name maple_oneoodprompt \
#             --network.backbone.PROMPT_DEPTH ${depth} \
#             --network.backbone.OOD_NUM ${OOD_NUM} \
#             --network.backbone.N_CTX ${N_CTX} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.interpolation bilinear \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 60 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./maple_mix/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_pdepth${depth} \
#             --seed ${seed}
#     done
# done




# #################################### ablate synimage
# mix_strategy=manimix_wccm
# depth=5
# alpha=1
# beta=1
# total_gs_num=1000
# loss_components=classify
# for gs_loss_weight in 1-1-1-1
# do
#     for train_data in train_syn_imagenet_sdxl_gs5 train_syn_imagenet_sdxl_sd3_gs5 train_syn_imagenet_sdxl_sd3_gs75
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/maple.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --dataset.train.imglist_pth ./data/benchmark_imglist/imagenet/train_imagenet.txt///./data/benchmark_imglist/imagenet/${train_data}.txt\
#             --network.name maple_oneoodprompt \
#             --network.backbone.PROMPT_DEPTH ${depth} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 50 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_synmix_imagenet/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_pdepth${depth}_${train_data} \
#             --seed 0
#     done
# done


# for gs_loss_weight in 1-1-1-1 1-1-1-0.1 1-1-1-0.01 1-1-1-5  
# do
#     for loss_components in abscos
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/maple.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name maple_oneoodprompt \
#             --network.backbone.PROMPT_DEPTH ${depth} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.loss_components ${loss_components} \
#             --trainer.trainer_args.gs_flag False \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 50 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_rerun_maple/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_pdepth${depth}_${loss_components}_lossw${gs_loss_weight} \
#             --seed 0
#     done
# done

# #################################### ablate gaussian sampling.
# mix_strategy=manimix_wccm
# depth=5
# alpha=1
# beta=1
# for total_gs_num in 100 300 1000 3000 
# do
#     for gs_loss_weight in 0.01 0.1 1 3
#     do
#         python main.py \
#             --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#             configs/networks/maple.yml \
#             configs/pipelines/train/train_coop.yml \
#             configs/preprocessors/base_preprocessor.yml \
#             configs/postprocessors/mcm.yml \
#             --network.name maple_oneoodprompt \
#             --network.backbone.PROMPT_DEPTH ${depth} \
#             --network.backbone.text_prompt tip \
#             --network.pretrained False \
#             --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#             --trainer.trainer_args.alpha ${alpha} \
#             --trainer.trainer_args.beta ${beta} \
#             --trainer.trainer_args.mix_strategy ${mix_strategy} \
#             --trainer.trainer_args.total_gs_num ${total_gs_num} \
#             --trainer.trainer_args.gs_loss_weight ${gs_loss_weight} \
#             --trainer.trainer_args.queue_capacity 500 \
#             --trainer.trainer_args.pre_queue True \
#             --trainer.trainer_args.iter_recompuration 10 \
#              --trainer.trainer_args.soft_split True \
#             --trainer.name oodmixup \
#             --postprocessor.name oneoodprompt \
#             --postprocessor.postprocessor_args.tau 1  \
#             --postprocessor.postprocessor_args.in_score sum  \
#             --dataset.train.batch_size 128 \
#             --dataset.train.few_shot 16 \
#             --optimizer.num_epochs 50 \
#             --optimizer.lr 0.1 \
#             --num_gpus 1 --num_workers 4 \
#             --merge_option merge \
#             --output_dir ./exp_gs_abla/ \
#             --mark alpha${alpha}_beta${beta}_${mix_strategy}_pdepth${depth}_total_gs_num${total_gs_num}_lossw${gs_loss_weight}_soft_split \
#             --seed 0
#     done
# done


# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/coop.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/msp.yml \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --seed 0
####################################################### mixup + coop.
# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/coop.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --trainer.trainer_args.alpha 0.2 \
#     --trainer.name mixup \
#     --postprocessor.postprocessor_args.tau 1  \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --output_dir ./prompt_tuning/ \
#     --mark coop_mixup \
#     --seed 0

# # ###################################################### baseline + maple.
# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/maple.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --trainer.trainer_args.beta 1.0 \
#     --trainer.trainer_args.cutmix_prob 1.0 \
#     --trainer.name base \
#     --postprocessor.postprocessor_args.tau 1  \
#     --network.backbone.PROMPT_DEPTH 9 \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --dataset.train.few_shot 16 \
#     --dataset.train.randseed 0 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --output_dir ./prompt_tuning/ \
#     --mark maple \
#     --seed 0

# ### mixup + maple | cutmix + maple.
# # ###################################################### cutmix + coop.
# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/maple.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --trainer.trainer_args.beta 1.0 \
#     --trainer.trainer_args.cutmix_prob 1.0 \
#     --trainer.name cutmix \
#     --postprocessor.postprocessor_args.tau 1  \
#     --network.backbone.PROMPT_DEPTH 9 \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --dataset.train.few_shot 16 \
#     --dataset.train.randseed 0 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --output_dir ./prompt_tuning/ \
#     --mark maple_cutmix \
#     --seed 0

# # ###################################################### cutmix + coop.
# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/maple.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --trainer.trainer_args.beta 1.0 \
#     --trainer.trainer_args.cutmix_prob 1.0 \
#     --trainer.name mixup \
#     --postprocessor.postprocessor_args.tau 1  \
#     --network.backbone.PROMPT_DEPTH 9 \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --dataset.train.few_shot 16 \
#     --dataset.train.randseed 0 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --output_dir ./prompt_tuning/ \
#     --mark maple_mixup \
#     --seed 0


# # ###################################################### cutmix + coop.
# python main.py \
#     --config configs/datasets/imagenet/imagenet_train_fsood.yml \
#     configs/networks/maple.yml \
#     configs/pipelines/train/train_coop.yml \
#     configs/preprocessors/base_preprocessor.yml \
#     configs/postprocessors/mcm.yml \
#     --trainer.trainer_args.beta 1.0 \
#     --trainer.trainer_args.cutmix_prob 1.0 \
#     --trainer.name cutmix \
#     --postprocessor.postprocessor_args.tau 1  \
#     --network.pretrained False \
#     --network.checkpoint ./results/pretrained_weights/resnet50_imagenet1k_v1.pth \
#     --dataset.train.batch_size 128 \
#     --optimizer.num_epochs 150 \
#     --num_gpus 1 --num_workers 6 \
#     --merge_option merge \
#     --output_dir ./prompt_tuning/ \
#     --mark maple \
#     --seed 0

