# ############################################################ 
## final results. AdaNeg on ImageNet
############################################################ 
#!/bin/bash
prompt=nice
in_score=far_only # ada or far_only or near_only // recommend use far_only for far-ood and near_only for near-ood, this will achieve more accurate results
random_permute=True
backbone=ViT-B/16
neglabel_init_flag=False
eta=0.50 # 0.75 0.50 0.25
# imagenet_traditional_four_ood
mllm_model_type=QWEN # LLAVA QWEN BLIP2 // Near-OOD must use LLAVA or QWEN, Far-OOD can use BLIP2 to accelerate
for group_num in 100  # 100 50 10
do
    for ens_stop_step in 10  # 20 10 5, use 5 will be more fast, use 20 get the best results
    do
        for datasetconfig in imagenet_traditional_four_ood
        do
            CUDA_VISIBLE_DEVICES=1 python main.py \
            --config configs/datasets/imagenet/${datasetconfig}.yml \
            configs/networks/fixed_clip.yml \
            configs/pipelines/test/test_fsood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/ants.yml \
            --dataset.test.batch_size 256 \
            --ood_dataset.batch_size 256 \
            --dataset.train.few_shot 0 \
            --dataset.num_classes 11000 \
            --ood_dataset.num_classes 11000 \
            --evaluator.name ood_clip_tta \
            --network.name fixedclip_negoodprompt \
            --network.backbone.ood_number 10000 \
            --network.backbone.name  ${backbone} \
            --network.backbone.text_prompt ${prompt} \
            --network.backbone.text_center True \
            --network.pretrained False \
            --postprocessor.APS_mode False \
            --postprocessor.name ants \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --postprocessor.postprocessor_args.in_score ${in_score}  \
            --postprocessor.postprocessor_args.neglabel_init_flag ${neglabel_init_flag}  \
            --postprocessor.postprocessor_args.eta ${eta}  \
            --postprocessor.postprocessor_args.ens_stop_step ${ens_stop_step}  \
            --postprocessor.postprocessor_args.mllm_model_type ${mllm_model_type}  \
            --num_gpus 1 --num_workers 8 \
            --merge_option merge \
            --output_dir ./cvpr_reimp/ \
            --mark ${datasetconfig}_${in_score}_beta${beta}_thres${thres}_gap${gap}_memleng${memleng}_lambda${lambdaval}_group_num_${group_num}_random_${random_permute}_ood
        done
    done    
done


