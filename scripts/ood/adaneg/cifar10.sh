# ############################################################ 
tau=1.0
beta=1.0
######################### AdaNeg, adopting a larger number of negative labels leads to better results.
prompt=nice 
##nice, cifar10 89.62.
##simple, 89.98.
##full: 90.74
random_permute=True  ### random group.
#  30 50 100 200 1000 2000asdf
group_num=1
thres=0.5

gap=0.0
in_score=sum
for ood_number in 70000 
do
    for thres in 0.0001 0.001 0.00001 0.01 
    do
        for gap in 0.0 0.1
        do 
            python main.py \
            --config configs/datasets/cifar10/cifar10_clip.yml \
            configs/networks/fixed_clip.yml \
            configs/pipelines/test/test_fsood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/mcm.yml \
            --dataset.train.batch_size 32 \
            --dataset.test.batch_size 32 \
            --ood_dataset.batch_size 32 \
            --dataset.train.few_shot 2 \
            --dataset.num_classes ${ood_number+10} \
            --evaluator.name ood_clip_tta \
            --network.name fixedclip_negoodprompt \
            --network.backbone.ood_number ${ood_number} \
            --network.backbone.text_prompt ${prompt} \
            --network.backbone.text_center True \
            --network.pretrained False \
            --postprocessor.APS_mode False \
            --postprocessor.name ttaprompt \
            --postprocessor.postprocessor_args.group_num ${group_num}  \
            --postprocessor.postprocessor_args.random_permute ${random_permute}  \
            --postprocessor.postprocessor_args.tau ${tau}  \
            --postprocessor.postprocessor_args.thres ${thres}  \
            --postprocessor.postprocessor_args.gap ${gap}  \
            --postprocessor.postprocessor_args.beta ${beta}  \
            --postprocessor.postprocessor_args.in_score ${in_score}  \
            --num_gpus 1 --num_workers 6 \
            --merge_option merge \
            --output_dir ./debug_nips2024/ \
            --mark ${in_score}_beta${beta}_thres_${thres}_gap_${gap}_negnum_${ood_number}_groupnum_${group_num}_random_${random_permute}_cifar10_tta
        done
    done
done
