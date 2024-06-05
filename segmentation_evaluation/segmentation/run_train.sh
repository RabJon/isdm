# !/bin/bash

export WANDB_API_KEY="PASTE_YOUR_PERSONAL_WANDB_API_KEY"

#FNOS=("R000FALL" "R100F000" "R100F100" "R100F200")
#FNOS=("R100F300" "R100F400" "R100F500")
#FNOS=("R100F600" "R100F700" "R100F800")
#FNOS=("R100F900" "R100F1000" "RALLF000")
#FNOS="RALLFALL"
#FNOS=("R000FALL" "R100F000" "R100F100" "R100F200" "R100F300" "R100F400" "R100F500" "R100F600" "R100F700" "R100F800" "R100F900" "R100F1000" "RALLF000" "RALLFALL")
FNOS=("R000FALL" "RALLF000" "RALLFALL" "R100F000" "R100F100" "R100F500" "R100F1000")


#model="UnetPlusPlus" # other options -> "DeepLabV3plus", "FPN", "UnetPlusPlus"
models=("UnetPlusPlus" "FPN" "DeepLabV3plus")
devices="[0]"

# for FNO in $FNOS
# do
#     config_file=my_configs/config_$FNO.yaml
#     exp_name=$model"_"$config_file
#     exp_dir=output/$exp_name
    
#     echo ======Running Training - $exp_name=======
#     python train.py fit --config=$config_file \
#                         --trainer.callbacks.init_args.dirpath=$exp_dir \
#                         --wandb_name=$exp_name \
#                         --trainer.callbacks.init_args.save_top_k=1 \
#                         --model.arch=$model \
#                         --trainer.devices $devices
    
#     echo ======Running Testing -  $exp_name=======
#     echo Running Testting - $config_file
#     python train.py test --config=$config_file \
#                          --trainer.callbacks.init_args.dirpath=$exp_dir \
#                          --wandb_name=$exp_name \
#                          --ckpt_path=$exp_dir/best.ckpt \
#                          --output_dir=$exp_dir \
#                          --model.arch=$model \
#                          --trainer.devices $devices
    
    
# done

for model in ${models[@]}
do
    for FNO in ${FNOS[@]}
    do
        # config_file=segmentation_experiments/my_configs/config_$FNO.yaml
        # config_file=segmentation_experiments/my_configs_all_latent/config_$FNO.yaml
        config_file=segmentation_experiments/my_configs_all_latent_corrected/config_$FNO.yaml
        exp_name=$model"_"$config_file
        #exp_dir=segmentation_experiments/output/$exp_name
        exp_dir=seg_exp7/output/$exp_name

        echo ======Running Training - $exp_name=======
        python segmentation_experiments/train.py fit --config=$config_file \
                            --trainer.callbacks.init_args.dirpath=$exp_dir \
                            --wandb_name=$exp_name \
                            --trainer.callbacks.init_args.save_top_k=1 \
                            --model.arch=$model \
                            --trainer.devices $devices
        
        echo ======Running Testing -  $exp_name=======
        echo Running Testting - $config_file
        python segmentation_experiments/train.py test --config=$config_file \
                            --trainer.callbacks.init_args.dirpath=$exp_dir \
                            --wandb_name=$exp_name \
                            --ckpt_path=$exp_dir/best.ckpt \
                            --output_dir=$exp_dir \
                            --model.arch=$model \
                            --trainer.devices $devices
    done
done