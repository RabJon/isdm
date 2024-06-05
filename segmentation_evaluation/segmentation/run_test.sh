# !/bin/bash

export WANDB_API_KEY="PASTE_YOUR_PERSONAL_WANDB_API_KEY"

FNOS=("R000FALL" "R100F000" "R100F100" "R100F200" "R100F300" "R100F400" "R100F500" "R100F600" "R100F700" "R100F800" "R100F900" "R100F1000" "RALLF000" "RALLFALL")
#model="DeepLabV3plus" # DeepLabV3plus #UnetPlusPlus #FPN
models=("UnetPlusPlus" "FPN" "DeepLabV3plus")
devices="[0]"

for model in ${models[@]}
do
    for FNO in ${FNOS[@]}
    do
        train_config_file=segmentation_experiments/my_configs_all_latent_corrected/config_$FNO".yaml"
        config_file=segmentation_experiments/my_configs_all_latent_corrected/config_$FNO"_test.yaml"
        exp_name=$model"_"$FNO
        ckpt_dir=seg_exp7/output/$model"_"$train_config_file
        out_dir=$ckpt_dir
        
        echo ======Run Testing -  $exp_name=======
        echo Running Testting - $config_file
        python segmentation_experiments/train.py test --config=$config_file \
                            --trainer.callbacks.init_args.dirpath=$exp_dir \
                            --wandb_name=$exp_name \
                            --ckpt_path=$ckpt_dir/best.ckpt \
                            --output_dir=$out_dir \
                            --model.arch=$model \
                            --trainer.devices $devices \
                            --model.test_print_num 0 \
                            --data.bs 20
        
    done
done