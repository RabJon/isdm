import argparse
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
from shutil import copyfile, rmtree, move
import filecmp
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np

from guided_diffusion.script_util import model_and_diffusion_defaults

import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

from image_train import train, get_training_defaults
from image_sample import sample, get_sampling_defaults

from balancer import balance


def read_config(config_path):
    config = None
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(config_path)
    
    return config

def save_config(config_path, config):
    with open(config_path, "w") as f:
        json.dump(config, f)


# class SystemCallError(Exception):
#     def __init__(self, command, exit_status):            
#         self.command = command
#         self.exit_status = exit_status
#         self.message = "System call error with exit status " + str(self.exit_status) + "when executing the command:\n" + str(self.command)
#         super().__init__(self.message)

# def make_system_call(command, check_exit_status = True):
#     print("Executing command:", command)
#     exit_status = os.system(command)
#     if check_exit_status and exit_status:
#         raise SystemCallError(command, exit_status)


def get_dataset_file_paths(train_root_path, test_root_path = None):
    def get_file_names(dir_path):
        
        if not os.path.isdir(dir_path):
            raise ValueError(str(dir_path) + " is not an existing folder!")
        
        images_subfolder = os.path.join(dir_path, "images")
        masks_subfolder = os.path.join(dir_path, "masks")
        if not os.path.isdir(images_subfolder):
            raise ValueError("The required images subfolder " +  str(images_subfolder), " is not an existing folder!")
        if not os.path.isdir(masks_subfolder):
            raise ValueError("The required masks subfolder " + str(masks_subfolder) + " is not an existing folder!")

        image_names_list = os.listdir(images_subfolder)
        image_names_list.sort()

        masks_names_list = os.listdir(masks_subfolder)
        masks_names_list.sort()

        if "Thumbs.db" in image_names_list:
            image_names_list.remove("Thumbs.db")
        
        if "Thumbs.db" in masks_names_list:
            masks_names_list.remove("Thumbs.db")

        for mask_name in masks_names_list:
            if  not mask_name.endswith(".png"):
                raise ValueError("Masks are required to have PNG format! Got file name " + mask_name)
            
        image_names_list_no_ext = [im.split(".")[0] for im in image_names_list]
        mask_names_list_no_ext = [m.split(".")[0] for m in masks_names_list]


        if len(image_names_list_no_ext) != len(mask_names_list_no_ext):
            raise ValueError("The images subfolder and the masks subfolder do not contain the same number of images! Each image should have exactly one corresponding mask!")

        if image_names_list_no_ext != mask_names_list_no_ext:
            raise ValueError("The image filenames and mask filenames are not the same!")
            

        return image_names_list, masks_names_list

    def check_duplicates(train_image_paths, test_image_paths):
        for train_path in train_image_paths:
            for test_path in test_image_paths:
                if filecmp.cmp(train_path, test_path, shallow=False):
                    raise ValueError("The files " + str(train_path) + " and " +  str(test_path) + " are identical.")
        return True

    train_image_file_names, train_mask_file_names = get_file_names(train_root_path)
    train_image_paths = [os.path.join(train_root_path, "images", n) for n in train_image_file_names]
    train_mask_paths = [os.path.join(train_root_path, "masks", n) for n in train_mask_file_names]
    train_file_paths = (train_image_paths, train_mask_paths)
    if test_root_path:
        test_image_file_names, test_mask_file_names = get_file_names(test_root_path)
        test_image_paths = [os.path.join(test_root_path, "images", n) for n in test_image_file_names]
        check_duplicates(train_image_paths, test_image_paths)
        test_mask_paths = [os.path.join(test_root_path, "masks", n) for n in test_mask_file_names]
        test_file_paths = (test_image_paths, test_mask_paths)

        return train_file_paths, test_file_paths
    
    
    return train_file_paths



def init_log_dir(action, config):
    
    dataset_name = config["data_dir"].split(os.sep)[-1]
    current_time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = current_time_stamp + "_" + action + "_" + dataset_name

    
    str_diffusion_steps = str(config["diffusion_steps"])
    str_num_classes = str(config["num_classes"])
    if action in ["train", "finetune", "sample"]:
        log_dir_name += "_d=" + str_diffusion_steps + "_c=" + str_num_classes
    
    if action == "finetune":
        str_drop_rate = str(config["drop_rate"])
        str_resume_checkpoint = str(config["resume_checkpoint"].split(os.sep)[-1])
        log_dir_name += "_dr=" + str_drop_rate + "_rc=" + str_resume_checkpoint
    elif action == "sample":
        str_num_samples = str(config["num_samples"])
        str_model_path = str(config["model_path"].split(os.sep)[-1])
        str_s = str(config["s"])
        log_dir_name += "_samples=" + str_num_samples + "_model=" + str_model_path + "_s=" + str_s
    
    log_dir = os.path.join(config["root_log_dir"], log_dir_name)

    # command = 'export OPENAI_LOGDIR="{}"'.format(log_dir)
    # make_system_call(command)
    os.makedirs(log_dir)

    return log_dir

def get_num_running_configs():
    root = os.path.dirname(__file__)
    running_configs = 0
    for file in os.listdir(root):
        if file.endswith("config_running.json"):
            running_configs += 1
    return running_configs

def main():
    print("CUDA available 1:", torch.cuda.is_available())
    
    # Argparser for this main
    parser = argparse.ArgumentParser(
        description="""Meta-program to run Semantic Diffusion experiments.""")
    parser.add_argument("action", choices=["train", "finetune", "sample", "test"], type=str, 
        help="""Controls which action the script should perform.""")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration file.", required=True)
    parser.add_argument("-s", "--seed", type=int, help="Random seed.", default=73)

    args = parser.parse_args()
    
    #Load the input config
    config_in = read_config(args.config)

    # Prepare the action
    if args.action == "train":

        if config_in["dataset_mode"] == "voc": #torchvision has a proper voc integration, nothing needs to be prepared
            print("Using PASCAL VOC Dataset")
            #These two need to be added as placeholders, such that the right dataloader is used
            train_file_paths = "train"
            val_file_paths = "trainval"
        else:
            image_file_paths, mask_file_paths = get_dataset_file_paths(os.path.join(config_in["data_dir"], "train"))
            train_images, val_images, train_masks, val_masks = train_test_split(image_file_paths, mask_file_paths, test_size=0.2, random_state=args.seed)
            train_file_paths = (train_images, train_masks)
            val_file_paths = (val_images, val_masks)
            print("Splitted dataset into", len(train_images), "training samples and", len(val_images), "validation samples.")
            
        # Add defaults to config
        config = get_training_defaults()
        config.update(model_and_diffusion_defaults())
        config.update(config_in)

        #Manipulate config to have better control over training procedure
        config["train_file_paths"] = train_file_paths
        config["val_file_paths"] = val_file_paths

    elif args.action == "sample":
        config = get_sampling_defaults()
        config.update(model_and_diffusion_defaults())
        config.update(config_in)

        #Manipulate config to have better control over sampling procedure
        if "balance_args" in config: #balancing is used
            image_file_paths, mask_file_paths = get_dataset_file_paths(os.path.join(config_in["data_dir"], "train"))
            balanced_indices = balance(mask_file_paths, config["num_samples"], config["num_classes"], height=config["image_size"], width=config["image_size"], **config["balance_args"])
            image_file_paths = np.array(image_file_paths)[balanced_indices]
            mask_file_paths = np.array(mask_file_paths)[balanced_indices]
            config["file_paths"] = (image_file_paths.tolist(), mask_file_paths.tolist())
        else:
            config["file_paths"] = None 


    
    
    #Prepare log and config files
    log_dir = init_log_dir(args.action, config)
    remove_log_dir_afterwards = False
    config["log_dir"] = log_dir
    print("Logging into:", log_dir)

    print("Running config for", args.action)
    #print(config)
    config_args = argparse.Namespace(**config)
    tmp_config_path = os.path.join(os.path.dirname(__file__), args.action + "_" + str(get_num_running_configs()) + "_config_running.json")
    save_config(tmp_config_path, vars(config_args))


    
    try:
        
        #Save the config
        if not (args.action == "sample"):
            save_config(os.path.join(log_dir, "input_config.json"), config_in)
            save_config(os.path.join(log_dir, args.action + "_config.json"), vars(config_args))
        else:
            save_config(os.path.join(config_args.results_path, "input_config.json"), config_in)
            save_config(os.path.join(config_args.results_path, args.action + "_config.json"), vars(config_args))

        
        #Run the action
        if args.action == "train":
            torch.cuda.empty_cache() 
            print("CUDA available 2:", torch.cuda.is_available())#print needed to have CUDA available
            train(config_args)
            
        elif args.action == "sample":
            remove_log_dir_afterwards = True
            config_args.results_path = os.path.join("RESULTS", os.path.basename(log_dir))
            print("CUDA available 2:", torch.cuda.is_available())#print needed to have CUDA available
            sample(config_args)
            

        elif args.action == "test":
            raise NotImplementedError(args.action + " action is not implemented!")
        
        
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_config_path)
        if remove_log_dir_afterwards:
            print("Removing", log_dir)
            rmtree(log_dir)

if __name__ == "__main__":
    main()