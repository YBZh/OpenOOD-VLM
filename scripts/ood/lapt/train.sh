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
        for weight_decay in 0.00001 
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