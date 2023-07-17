import argparse
from genericpath import isdir
import os
import sys
import random
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("/".join(current_dir.split('/')[:-2]))
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import constants.constants as constants
from torch_dataset import TestSet
from utils.model_utils import find_model

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def getData(path_data_out, features = None):
    test_set = TestSet()
    gened_seqs = []
    real_seqs = []
    for file in os.listdir(path_data_out):
        pd_file = pd.read_csv(path_data_out + file)
        pd_file = pd_file[["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",\
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]]
        if features != None:
            pd_file = pd_file[features]
        gened_seqs.append(pd_file)

    if features != None:
        for test_video in test_set.Y_final_ori:
            real_seqs.append(test_video[features])
    else:
        real_seqs = test_set.Y_final_ori

    gened_frames = np.concatenate(gened_seqs, axis=0)
    real_frames = np.concatenate(real_seqs, axis=0)
    return real_frames, gened_frames


def create_pca(real_frames, gened_frames, pdf, features_name = ""):
    scaler = StandardScaler()
    scaler.fit(real_frames)
    X_gened = scaler.transform(gened_frames) 
    X_real = scaler.transform(real_frames) 

    mypca = PCA(n_components=2, random_state = 1) # On paramètre ici pour ne garder que 2 axes
    mypca.fit(X_real)

    print(mypca.singular_values_) # Valeurs de variance
    print('Explained variation per principal component: {}'.format(mypca.explained_variance_ratio_))
    data_generated = mypca.transform(X_gened)
    data_real = mypca.transform(X_real)

    df_generated = pd.DataFrame(data = data_generated, columns = ['principal component 1', 'principal component 2'])
    df_real = pd.DataFrame(data = data_real, columns = ['principal component 1', 'principal component 2'])

    indicesToKeep = random.sample(range(len(df_generated)), 1000)

    plt.figure(figsize=(3, 3), dpi=100)
    plt.title('pca_'+features_name)
    plt.scatter(df_real.loc[indicesToKeep, 'principal component 1'], df_real.loc[indicesToKeep, 'principal component 2'], label='Real data', rasterized=True)
    plt.scatter(df_generated.loc[indicesToKeep, 'principal component 1'], df_generated.loc[indicesToKeep, 'principal component 2'], label='Generated data', alpha=0.7, rasterized=True)
    plt.xlabel('Principal Component - 1')
    plt.ylabel('Principal Component - 2')
    plt.legend()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
def calculate_kde(real_frames, gened_frames, path_evaluation, features_name = "", bandwidth = None):
    if(bandwidth == None):
        params = {'bandwidth':  np.logspace(-2, 0, 5)}
        print("Grid search for bandwith parameter of Kernel Density...")
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(gened_frames)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        scores = grid.best_estimator_.score_samples(real_frames)

    else:
        kde = KernelDensity(kernel='gaussian', bandwidth = float(bandwidth)).fit(gened_frames)
        scores = kde.score_samples(real_frames)

    mean = np.mean(scores)
    sd = np.std(scores)
    print("mean ", str(mean))
    print("ses ", str(sd))

    return mean, sd

def plot_figure(real_signal, generated_signal, pdf, features_name):
        x_real = range(len(real_signal))
        x_gen = range(len(generated_signal))
        plt.figure(figsize=(3, 3), dpi=100)
        plt.title(features_name)
        plt.plot(x_gen, generated_signal, label="generated", alpha=0.5, rasterized=True)
        plt.plot(x_real, real_signal, label="real", alpha=0.8, rasterized=True)
        plt.legend()
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-params', help='Path to the constant file', default="./params.cfg")
    parser.add_argument('-epoch', help='number of epoch of recovred model', default=9)
    parser.add_argument('-bandwidth', help= "bandwith in known", default=None)
    parser.add_argument('-kde', action='store_true')
    parser.add_argument('-pca', action='store_true')
    args = parser.parse_args()
    constants = read_params(args.params)

    model_file = find_model(int(args.epoch)) 

    path_data_out = constants.output_path + model_file[0:-3] + "/"
    if(not isdir(path_data_out)):
        raise Exception(path_data_out + "is not a directory")

    path_evaluation = constants.evaluation_path + model_file[0:-3] + "/"
    print(path_evaluation)

    if(not isdir(path_evaluation)):
        os.makedirs(path_evaluation, exist_ok=True)

    #rajouter pour chaque features juste un graphe avec données réel et données générée (valeur en fonction du temps ??)
    all_features = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry",
                "pose_Rz", "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
    
    with PdfPages(path_evaluation + "curve.pdf") as pdf:
        for feature in all_features : 
            real_frames, gened_frames = getData(path_data_out, feature)
            plot = plot_figure(real_frames, gened_frames, pdf, feature)

    #creation of different mesures for each type of output
    types_output = {
    "first_eye" : ["gaze_0_x", "gaze_0_y", "gaze_0_z"],
    "second_eye" : ["gaze_1_x", "gaze_1_y", "gaze_1_z"],
    "gaze_angle" : ["gaze_angle_x", "gaze_angle_y"],
    "pose" : ["pose_Rx", "pose_Ry", "pose_Rz"],
    "sourcils" : ["AU01_r", "AU02_r", "AU04_r"],
    "visage" : ["AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r"],
    "bouche" : ["AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r"],
    "clignement" : ["AU45_r"]}

    if(args.pca or args.kde):
        df_kde = pd.DataFrame()  
        mean_tab = []
        sd_tab = []
        with PdfPages(path_evaluation + "PCA.pdf") as pdf:
            for cle, valeur in types_output.items():
                real_frames, gened_frames = getData(path_data_out, valeur)
                if(cle != "clignement" and args.pca):
                    print("create pca for", cle)
                    pca = create_pca(real_frames, gened_frames, pdf, cle)
                if(args.kde):
                    print("calculate kde for", cle)
                    mean, sd = calculate_kde(real_frames, gened_frames, path_evaluation, cle, args.bandwidth)   
                    mean_tab.append(mean)
                    sd_tab.append(sd)
            if(args.kde):
                data = {"mean":mean_tab, "sd": sd_tab}
                df_kde = pd.DataFrame(data, index=types_output.keys())
                df_kde.to_csv(path_evaluation + "eval.csv")