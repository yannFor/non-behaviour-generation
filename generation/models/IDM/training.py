
from gaussianDiffusion import *
from sampler import create_named_schedule_sampler
from model import UNet_conditional
from Unet1D import Unet1D
import torch
import os,sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_dir.split('/')[:-2]))
from constants import constants,constants_utils
from models.TrainClass import Train
from train_util import TrainLoop

import argparse

import configparser
config = configparser.RawConfigParser()



def read_params(file):

    config.read(file)
    
    # --- model type params
    constants.kernel_size = config.getint('MODEL_TYPE','kernel_size')
    constants.dropout =  config.getfloat('MODEL_TYPE','dropout') 
    datasets = config.get('PATH','datasets')
    constants.datasets = datasets.split(",")
    constants.dir_path = config.get('PATH','dir_path')
    constants.data_path = config.get('PATH','data_path')
    constants.saved_path = config.get('PATH','saved_path')
    constants.output_path = config.get('PATH','output_path')
    constants.evaluation_path = config.get('PATH','evaluation_path')
    constants.model_path = config.get('PATH','model_path')

    # --- Training params
    constants.n_epochs =  config.getint('TRAIN','n_epochs')
    constants.batch_size = config.getint('TRAIN','batch_size')
    constants.lr =  config.getfloat('TRAIN','lr')
    constants.log_interval =  config.getint('TRAIN','log_interval')
    constants.learn_sigma = config.getboolean('TRAIN','learn_sigma')
    constants.diffusion_steps = config.getint('TRAIN','diffusion_steps')
    constants.noise_schedule = config.get('TRAIN','noise_schedule')
    constants.use_kl = config.getboolean('TRAIN','use_kl')
    constants.rescale_timesteps = config.getboolean('TRAIN','rescale_timesteps')
    constants.rescale_learned_sigmas = config.getboolean('TRAIN','rescale_learned_sigmas')
    constants.timestep_respacing = config.getboolean('TRAIN','timestep_respacing')
    constants.sigma_small = config.getboolean('TRAIN','sigma_small')
    constants.predict_xstart = config.getboolean('TRAIN','predict_xstart')
    constants.schedule_sampler = config.get('TRAIN','schedule_sampler')
    constants.ema_rate = config.getfloat('TRAIN','ema_rate')
    constants.scheduler_factor = config.getfloat('TRAIN','scheduler_factor')

    # --- Data params
    constants.noise_size = config.getint('DATA','noise_size')
    constants.pose_size = config.getint('DATA','pose_size') 
    constants.eye_size = config.getint('DATA','eye_size')
    constants.pose_t_size = config.getint('DATA','pose_t_size')
    constants.pose_r_size = config.getint('DATA','pose_r_size')
    constants.au_size = config.getint('DATA','au_size') 
    constants.derivative = config.getboolean('DATA','derivative')

    constants.opensmile_columns = get_config_columns('opensmile_columns')
    constants.selected_opensmile_columns = get_config_columns('selected_opensmile_columns')

    constants.openface_columns = get_config_columns('openface_columns')
    constants.features_size =  config.getint('DATA',"features_size")
    constants.prosody_size = config.getint('DATA',"prosody_size")

    base_size = len(constants.selected_opensmile_columns) 

    if constants.derivative:
        constants.prosody_size = base_size * 3
    else:
        constants.prosody_size = base_size

    constants.selected_os_index_columns = []
    for column in constants.selected_opensmile_columns:
        constants.selected_os_index_columns.append(constants.opensmile_columns.index(column))
    constants.saved_path = create_saved_path(file)
    return constants


def get_config_columns(list):
    list_items = config.items(list)
    columns = []
    for _, column_name in list_items:
        columns.append(column_name)
    return columns

def create_saved_path(config_file):
    # * Create dir for store models, hist and params
    str_dataset = ""
    for dataset in constants.datasets:
        str_dataset += dataset + "_"

    saved_path = constants.saved_path + str_dataset + constants.model_path
    os.makedirs(saved_path, exist_ok=True)

    return saved_path




def main():
    #args = create_argparser().parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    args = parser.parse_args()
    constants = read_params(args.params)


    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(constants)

    if torch.cuda.is_available():
        device =  torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    schedule_sampler = create_named_schedule_sampler(str(constants.schedule_sampler), diffusion)

    print("creating data loader...")
    
    trainClass = Train()
    trainLoader = trainClass.trainloader
    testLoader = trainClass.testloader 

    TrainLoop(
       model=model,
       diffusion=diffusion,
       data=[trainLoader,testLoader],
       batch_size=constants.batch_size,
       lr=constants.lr,
       ema_rate=constants.ema_rate,
       schedule_sampler=schedule_sampler,
       weight_decay=0.0,
       save_interval = constants.log_interval,
       epochs = constants.n_epochs,
       saved_path = constants.saved_path,
       scheduler_factor=constants.scheduler_factor
    ).run_loop()
    

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser









def create_model_and_diffusion(constants):
    model = create_model(
        out_channels=constants.features_size,
        num_channels=constants.features_size,
        learn_sigma=constants.learn_sigma,
        kernel_size = constants.kernel_size,
        prosody_size = constants.prosody_size
    )
    diffusion = create_gaussian_diffusion(
        steps=constants.diffusion_steps,
        learn_sigma=constants.learn_sigma,
        sigma_small=constants.sigma_small,
        noise_schedule=constants.noise_schedule,
        use_kl=constants.use_kl,
        predict_xstart=constants.predict_xstart,
        rescale_timesteps=constants.rescale_timesteps,
        rescale_learned_sigmas=constants.rescale_learned_sigmas,
        timestep_respacing=constants.timestep_respacing,
    )
    return model, diffusion


def create_model(
    out_channels,
    num_channels,
    learn_sigma,
    kernel_size,
    prosody_size,
    num_res_blocks=None,
    class_cond=None,
    use_checkpoint=None,
    attention_resolutions=None,
    num_heads=None,
    num_heads_upsample=None,
    use_scale_shift_norm=None,
    dropout=None,
):

    out_channels = (num_channels if not learn_sigma else num_channels * 2)
    return UNet_conditional(num_channels=num_channels,input_dim = prosody_size, c_out=out_channels,kernel_size=kernel_size,device=("cuda:0" if torch.cuda.is_available() else 'cpu'))
    #return Unet1D(dim=128,channels=num_channels + prosody_size, learned_variance = learn_sigma)









def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return GaussianDiffusion(betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )



if __name__ == "__main__":
    main()






