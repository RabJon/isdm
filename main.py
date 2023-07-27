import argparse
import json
import os
from shutil import copyfile, rmtree, move

from guided_diffusion.script_util import model_and_diffusion_defaults

import torch

from image_train import train
from image_sample import sample, get_sampling_defaults


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


def init_log_dir(action, config):
    
    dataset_name = config["data_dir"].split(os.sep)[-1]
    log_dir_name = action + "_" + dataset_name

    
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
    
    parser = argparse.ArgumentParser(
        description="""Meta-program to run Semantic Diffusion experiments.""")
    
    parser.add_argument("action", choices=["train", "finetune", "sample", "test"], type=str, 
        help="""Controls which action the script should perform.""")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration file.", required=True)

    
    args = parser.parse_args()

    config_in = read_config(args.config)
    if args.action == "sample":
        config = get_sampling_defaults()
        config.update(model_and_diffusion_defaults())
        config.update(config_in)

    log_dir = init_log_dir(args.action, config)
    remove_log_dir_afterwards = False
    config["log_dir"] = log_dir
    print("Logging into:", log_dir)

    config_args = argparse.Namespace(**config)
    
    tmp_config_path = os.path.join(os.path.dirname(__file__), args.action + "_" + str(get_num_running_configs()) + "_config_running.json")
    save_config(tmp_config_path, vars(config_args))

    try:

        if args.action in ["train", "finetune"]:
            train(config_args)
            
        elif args.action == "sample":
            remove_log_dir_afterwards = True
            config_args.results_path = os.path.join("RESULTS", os.path.basename(log_dir))
            print("CUDA available 2:", torch.cuda.is_available())#print needed to have CUDA available
            sample(config_args)
            

        elif args.action == "test":
            raise NotImplementedError(args.action + " action is not implemented!")
        
        if not (args.action == "sample"):
            save_config(os.path.join(log_dir, "input_config.json"), config_in)
            save_config(os.path.join(log_dir, args.action + "_config.json"), vars(config_args))
        else:
            save_config(os.path.join(config_args.results_path, "input_config.json"), config_in)
            save_config(os.path.join(config_args.results_path, args.action + "_config.json"), vars(config_args))
    except Exception as e:
        raise e
    finally:
        os.remove(tmp_config_path)
        if remove_log_dir_afterwards:
            print("Removing", log_dir)
            rmtree(log_dir)

if __name__ == "__main__":
    main()