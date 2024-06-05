import os
import re
import yaml
from shutil import copyfile, rmtree, move
import numpy as np


class SystemCallError(Exception):
    def __init__(self, command, exit_status):            
        self.command = command
        self.exit_status = exit_status
        self.message = "System call error with exit status " + str(self.exit_status) + "when executing the command:\n" + str(self.command)
        super().__init__(self.message)


def make_system_call(command, check_exit_status = True):
    print("Executing command:", command)
    exit_status = os.system(command)
    if check_exit_status and exit_status:
        raise SystemCallError(command, exit_status)
    

def manipulate_base_segmentation_model_config(log_folder, base_config_path, config_code, real_dataset_paths, generated_dataset_paths = None, num_classes = 2, image_size=256, is_evaluation=False):
    
    def try_cast_to_int(value):
        try:
            int_value = int(value)
            return int_value
        except:
            return value
    
    base_config = {}
    #base_ldm_config_path = "./latent-diffusion-base-configs/masks/00.yaml"
    with open(base_config_path, "r") as stream:
        try:
            base_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    code = re.search("R(.*)F", config_code)
    if code is not None:
        code = code.group(1)
    else:
        raise ValueError("Could not parse config code:", config_code)

    num_real_samples = try_cast_to_int(code)
    if num_real_samples == "ALL":
        num_real_samples = -1
    num_fake_samples = try_cast_to_int(config_code.split("F")[1])
    if num_fake_samples == "ALL":
        num_fake_samples = -1
    
    classes = list(range(num_classes))
    base_config["data"]["data"]["classes"] = classes
    base_config["data"]["data"]["image_size"] = image_size
    # if num_classes != 2:
    #     base_config["model"]["out_classes"] = num_classes
    base_config["model"]["out_classes"] = num_classes
    if not is_evaluation:
        base_config["data"]["data"]["train"][0]["img_dir"] = real_dataset_paths["train"]["image"]
        base_config["data"]["data"]["train"][0]["mask_dir"] = real_dataset_paths["train"]["mask"]
        base_config["data"]["data"]["train"][0]["num_samples"] = num_real_samples
        # base_config["data"]["data"]["train"][0]["classes"] = classes #TODO: think about moving this param one level up in config, because it must be same for train and validation sets

        base_config["data"]["data"]["train"][1]["img_dir"] = generated_dataset_paths["train"]["image"]
        base_config["data"]["data"]["train"][1]["mask_dir"] = generated_dataset_paths["train"]["mask"]
        base_config["data"]["data"]["train"][1]["num_samples"] = num_fake_samples
        # base_config["data"]["data"]["train"][1]["classes"] = classes

        base_config["data"]["data"]["validation"][0]["img_dir"] = real_dataset_paths["val"]["image"]
        base_config["data"]["data"]["validation"][0]["mask_dir"] = real_dataset_paths["val"]["mask"]
        # base_config["data"]["data"]["validation"][0]["classes"] = classes
    else:
        base_config["data"]["data"]["validation"][0]["img_dir"] = real_dataset_paths["test"]["image"]
        base_config["data"]["data"]["validation"][0]["mask_dir"] = real_dataset_paths["test"]["mask"]
        # base_config["data"]["data"]["validation"][0]["classes"] = classes


    # base_config["trainer"]["max_epochs"] = 100 #TODO: delete this later (just test if better to use more epochs!)

    config_path = os.path.join(log_folder + "/tmp", "segmentation_config_" + config_code + ".yaml")
    with open(config_path, 'w') as file:
        yaml.dump(base_config, file)
    
    return config_path


def train_segmentation_models(log_folder, base_config_path, real_dataset_paths, generated_dataset_paths, num_classes = 2, image_size = 256, test_codes=None, models=None):

    #Ensure that a new folder is created for each experiment
    segmentation_folders = [
        f for f in os.listdir(log_folder) if os.path.isdir(os.path.join(log_folder, f)) and "segmentation_experiment_" in f]
    experiment_id = len(segmentation_folders)
    experiment_base_folder = os.path.join(log_folder, "segmentation_experiment_" + str(experiment_id))
    while os.path.isdir(experiment_base_folder):
        experiment_id += 1
        experiment_base_folder = os.path.join(log_folder, "segmentation_experiment_" + str(experiment_id))

    if not models:
        models=("UnetPlusPlus", "FPN", "DeepLabV3plus")
    # models=("UnetPlusPlus",)
    # models=("FPN", "DeepLabV3plus")
    
    #FNOS=("R000FALL", "RALLF000", "RALLFALL", "R100F000", "R100F100", "R100F500", "R100F1000")
    num_real_train_data = len([f for f in os.listdir(real_dataset_paths["train"]["image"]) if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))])
    
    if not test_codes: #use default
        fno_fake_instead_of_real = "R000F" + str(num_real_train_data)
        fno_same_fake_same_real = "R" + str(num_real_train_data) + "F" + str(num_real_train_data)
        FNOS=("RALLF000", "RALLF100", "RALLF400", "RALLF700" , "RALLFALL", fno_fake_instead_of_real, fno_same_fake_same_real, "R000FALL")
    else:
        FNOS = test_codes

    # FNOS = ("R000FALL",)
    devices="[0]"


    for model in models:
        for fno in FNOS:
            # experiment_name = model + "_" + fno
            experiment_name = model + "/" + fno
            print("======Running Training - {}=======".format(experiment_name))
            experiment_folder = os.path.join(experiment_base_folder, experiment_name)

            config_path = manipulate_base_segmentation_model_config(log_folder, base_config_path, fno, real_dataset_paths, generated_dataset_paths, num_classes = num_classes, image_size = image_size)
            #run_train call
            # command = 'export WANDB_API_KEY="bbc1b23a1393868a8e9319415e74e3eebc6c02a5";python segmentation/train.py fit'
            command = 'python segmentation/train.py fit'
            arguments = [
                " --config " + config_path,
                " --trainer.callbacks.init_args.dirpath=" +  experiment_folder,
                " --wandb_name=" + experiment_name,
                " --trainer.callbacks.init_args.save_top_k=1 ",
                " --model.arch=" + model,
                " --trainer.devices " + devices
            ]

            command += " ".join(arguments)

            make_system_call(command)
            
            move(config_path, os.path.join(experiment_folder, "config_train.yaml"))




def evaluate_segmentation_models(log_folder, base_config_path, experiment_id, real_dataset_paths, num_classes = 2, test_codes=None, models=None):
        
    
    if experiment_id == -1:
        experiment_folder_ids = [int(f.split("_")[-1]) for f in os.listdir(log_folder) if "segmentation_experiment_" in f]
        experiment_id = max(experiment_folder_ids)
    
    experiment_folder = os.path.join(log_folder, "segmentation_experiment_" + str(experiment_id))
    if not os.path.isdir(experiment_folder):
        raise FileNotFoundError("Could not find experiment folder:", experiment_folder)

    if not models:
        models=("UnetPlusPlus", "FPN", "DeepLabV3plus")
    # models=("UnetPlusPlus",)
    # models=("FPN", "DeepLabV3plus")
    #FNOS=("R000FALL", "RALLF000", "RALLFALL", "R100F000", "R100F100", "R100F500", "R100F1000")
    num_real_train_data = len([f for f in os.listdir(real_dataset_paths["train"]["image"]) if (f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg"))])
    
    if not test_codes: #use default
        fno_fake_instead_of_real = "R000F" + str(num_real_train_data)
        fno_same_fake_same_real = "R" + str(num_real_train_data) + "F" + str(num_real_train_data)
        FNOS=("RALLF000", "RALLF100", "RALLF400", "RALLF700" , "RALLFALL", fno_fake_instead_of_real, fno_same_fake_same_real, "R000FALL")
    else:
        FNOS = test_codes
    
    
    devices="[0]"

    for model in models:
        for fno in FNOS:
            # experiment_name = model + "_" + fno
            experiment_name = model + "/" + fno
            print("======Running Testing - {}=======".format(experiment_name))

            config_path = manipulate_base_segmentation_model_config(log_folder, base_config_path, fno, real_dataset_paths, num_classes = num_classes, is_evaluation=True)
            #run_train call
            # command = 'export WANDB_API_KEY="bbc1b23a1393868a8e9319415e74e3eebc6c02a5";python segmentation/train.py test'
            command = 'python segmentation/train.py test'
            arguments = [
                " --config " + config_path,
                " --trainer.callbacks.init_args.dirpath=" +  experiment_folder,
                " --wandb_name=" + experiment_name,
                " --ckpt_path=" + experiment_folder + "/" + experiment_name + "/best.ckpt",
                " --output_dir=" + experiment_folder + "/" + experiment_name,
                " --model.arch=" + model,
                " --trainer.devices " + devices,
                " --model.test_print_num 10 ",
                " --data.bs 20 "
            ]

            command += " ".join(arguments)

            make_system_call(command)
            
            move(config_path, os.path.join(experiment_folder, "config_test.yaml"))


def prepare_real_train_data(tmp_folder, train_set_folder, dataset_txt_paths):
    
    def copy_files(srcs, dst_folder):
        os.makedirs(dst_folder)
        for src in srcs:
            name = os.path.basename(src)
            copyfile(src, os.path.join(dst_folder, name))
    
    
    train_image_file_paths = [os.path.join(train_set_folder, "images", fname) for fname in np.loadtxt(dataset_txt_paths["train"]["images"], dtype=str)]
    val_image_file_paths = [os.path.join(train_set_folder, "images", fname) for fname in np.loadtxt(dataset_txt_paths["val"]["images"], dtype=str)]
    train_mask_file_paths = [os.path.join(train_set_folder, "masks", fname) for fname in np.loadtxt(dataset_txt_paths["train"]["masks"], dtype=str)]
    val_mask_file_paths = [os.path.join(train_set_folder, "masks", fname) for fname in np.loadtxt(dataset_txt_paths["val"]["masks"], dtype=str)]

    
    
    train_image_folder = os.path.join(tmp_folder, "train", "images")
    train_mask_folder = os.path.join(tmp_folder, "train", "masks")
    val_image_folder = os.path.join(tmp_folder, "val", "images")
    val_mask_folder = os.path.join(tmp_folder, "val", "masks")
    copy_files(train_image_file_paths, train_image_folder)
    copy_files(train_mask_file_paths, train_mask_folder)
    copy_files(val_image_file_paths, val_image_folder)
    copy_files(val_mask_file_paths, val_mask_folder)

    dataset_paths = {
            "train": {"image": train_image_folder, "mask": train_mask_folder},
            "val": {"image": val_image_folder, "mask": val_mask_folder}}
    
    return dataset_paths



def main():

    #TODO: read config file and create args_dict from it
    
    real_dataset_paths = prepare_real_train_data(tmp_folder, args_dict["train"], dataset_txt_paths)
    real_dataset_paths["test"] = {
        "image": os.path.join(args_dict["test"], "images"),
        "mask": os.path.join(args_dict["test"], "masks")}
    
    generated_dataset_paths = None
    if args_dict["action"] != "eval":#in eval mode we only need real test data, no generated data
        generated_masks_folder = None
        if (args_dict["action"] != "all") and ("generated_masks" in args_dict):
            generated_masks_folder = args_dict["generated_masks"]
        else:
            if args_dict["semantic_image_synthesis"]:
                generated_masks_folder = os.path.join(log_folder, "SIS_generated_masks")
            else:
                generated_masks_folder = os.path.join(log_folder, "generated_masks", "img")
            if not (os.path.isdir(generated_masks_folder) and len(os.listdir(generated_masks_folder)) >= args_dict["num_samples"]):
                raise ValueError("'generated_masks' folder was neither provided nor retrieved in" + log_folder)
            print("Using generated masks in:", generated_masks_folder)
            
        generated_images_folder = None
        if (args_dict["action"] != "all") and ("generated_images" in args_dict):
            generated_images_folder = args_dict["generated_images"]
        else:
            generated_images_folder = os.path.join(log_folder, "generated_images", "img")
            if not (os.path.isdir(generated_images_folder) and len(os.listdir(generated_images_folder)) >= args_dict["num_samples"]):
                raise ValueError("'generated_images' folder was neither provided nor retrieved in" + log_folder)
            print("Using generated images in:", generated_images_folder)
        
        generated_dataset_paths = {
            "train": {
                "image": generated_images_folder, 
                "mask": generated_masks_folder}
        }

    test_codes = None if not "test_codes" in args_dict else args_dict["test_codes"]
    segmentation_models = None if not "models" in args_dict else args_dict["models"]
    if args_dict["action"] != "eval":
        refuse_log_folder_removing = True
        train_segmentation_models(log_folder, base_seg_model_config_path, real_dataset_paths, generated_dataset_paths, num_classes = args_dict["num_classes"], test_codes=test_codes, models = segmentation_models)
        print("Training of the Segmentation models terminated. Starting to test them!")
    if args_dict["action"] != "TSEG":
        evaluate_segmentation_models(
            log_folder, base_seg_model_config_path, args_dict["segmentation_experiment_id"], real_dataset_paths, num_classes = args_dict["num_classes"], test_codes=test_codes, models = segmentation_models)
        print("Testing of the Segmentation models terminated!")

if __name__ == '__main__':
    main()