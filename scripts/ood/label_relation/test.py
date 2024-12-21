
# ############################################################ 
tau=1.0
beta=1.0
######################### clip + ood + NegLabel
prompt=nice
random_permute=False
in_score=sum
group_num=1
ood_number=1000
for beta in 3
do
    for in_score in mul_mean
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
        --output_dir ./dug_not_used/ \
        --mark ${in_score}_beta${beta}_neg10k_ood_number_${ood_number}_random_${random_permute}
    done
done

