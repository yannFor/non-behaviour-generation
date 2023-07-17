import argparse
from genericpath import isdir
import os,sys
import numpy as np
import torch
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_dir.split('/')[:-2]))
import constants.constants as constants
from utils.create_final_file import createFinalFile
from torch_dataset import TestSet,TrainSet
import pandas as pd
from model import UNet_conditional
from training import create_gaussian_diffusion
from Unet1D import Unet1D

import configparser
config = configparser.RawConfigParser()

def get_config_columns(list):
    list_items = config.items(list)
    columns = []
    for _, column_name in list_items:
        columns.append(column_name)
    return columns

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

    return constants



def find_model(epoch):
    model_file = constants.model_path + "epoch"
    model_file += f"_{epoch}.pt"
    return model_file

def load_model(param_path, device):
    out_channels = (constants.features_size if not constants.learn_sigma else constants.features_size * 2)
    model = UNet_conditional(num_channels=constants.features_size,input_dim = constants.prosody_size, c_out=out_channels,kernel_size=constants.kernel_size,device="cuda:0")
    #model = Unet1D(dim=64,channels=constants.features_size + constants.prosody_size, learned_variance = constants.learn_sigma)
    model.load_state_dict(torch.load(param_path, map_location=device))
    return model.to(device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-dataset', help='wich video we want to generate', default="")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    constants = read_params(args.params)

    if(args.dataset != ""):
        datasets = args.dataset
        constants.datasets = datasets.split(",")

    model_file = find_model(int(args.epoch)) 
    #print(constants.saved_path + model_file)
    #model = load_model(constants.saved_path + model_file, device)


    path_data_out = constants.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        os.makedirs(path_data_out, exist_ok=True)

    trainset = TrainSet()
    trainset.scaling(True)
    y_scaler = trainset.y_scaler

    testset = TestSet()
    testset.scaling(trainset.x_scaler, trainset.y_scaler)
    

    gened_seqs = []
    columns = constants.openface_columns

    current_part = 0 
    current_key = ""
    df_list = []

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

    #model.eval()
    #sample_fn = diffusion.p_sample_loop
    for index, data in enumerate(testset, 0):
        input, target = torch.tensor(data[0]).float(), torch.tensor(data[1]).float()
        input = input.float().swapaxes(0,1)
        key = trainset.getInterval(index)[0]
        if(current_key != key): #process of a new video
            if(current_key != ""):
                createFinalFile(path_data_out, current_key, df_list)
            print("Generation of video", key, "...")
            current_part = 0
            current_key = key
            df_list = []
        cond = input.reshape(-1,input.shape[0], input.shape[1])
        if current_part < 3:
            #sample = sample_fn(model,(1,28,100),cond)
            #sample = sample.clamp(0,1)
            #sample = sample.swapaxes(1,2).squeeze(0).cpu()
            #out = y_scaler.inverse_transform(sample)
            out = y_scaler.inverse_transform(target)
            print(out.shape)
            #add timestamp and head translation for greta
            timestamp = np.array(trainset.getInterval(index)[1][:,0])
            out = np.concatenate((timestamp.reshape(-1,1), out[:,:constants.eye_size], np.zeros((out.shape[0], 3)), out[:,constants.eye_size:]), axis=1)
            df = pd.DataFrame(data = out, columns = columns)
            df_list.append(df)
            current_part += 1

    createFinalFile(path_data_out, current_key, df_list)


        

