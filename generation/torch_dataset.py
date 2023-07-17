import numpy as np
import pickle
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import constants.constants as constants
import speechpy

class Set(Dataset):

    def __init__(self, setType = "train"):
        # Load data
        path = constants.data_path
        datasets = constants.datasets
        X = []
        Y = []
        interval = []

        for set_name in datasets:
            current_X = []
            with open(path +'X_'+setType+'_'+set_name+'.p', 'rb') as f:
                x = pickle.load(f)
            current_X = np.array(x)[:,:,np.r_[constants.selected_os_index_columns]]

            if constants.derivative:
                current_X = self.addDerevative(current_X)


            with open(path +'y_'+setType+'_'+set_name+'.p', 'rb') as f:
                y = pickle.load(f)
                #current_Y = np.array(y)[:,:,constants.eye_size:constants.eye_size + 3]
                current_Y = np.array(y)

            with open(path +'intervals_test_'+set_name+'.p', 'rb') as f:
                current_interval = pickle.load(f)

            X.extend(current_X)
            Y.extend(current_Y)
            interval.extend(current_interval)
        
        X_concat, Y_concat = self.flatten(X, Y)

        x_scaler = MinMaxScaler((-1,1)).fit(X_concat) 
        y_scaler = MinMaxScaler((-1,1)).fit(Y_concat)
        #x_scaler = StandardScaler().fit(X_concat)
        #y_scaler = StandardScaler().fit(Y_concat)

        X_scaled_concat = x_scaler.transform(X_concat)
        Y_scaled_concat = y_scaler.transform(Y_concat)

        X_scaled, Y_scaled = self.reshape(X_scaled_concat, Y_scaled_concat)

        self.X = X
        self.Y = Y
        self.interval = interval
        self.X_scaled = X_scaled
        self.Y_scaled = Y_scaled
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        if(setType == "test"):
            with open(path +'y_test_final_'+set_name+'.p', 'rb') as f:
                self.Y_final_ori = pickle.load(f)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

    def getInterval(self, i):
        return self.interval[i]
    
    def flatten(self, X, Y):
        X_concat = np.reshape(X,(-1, constants.prosody_size))
        Y_concat = np.reshape(Y,(-1, constants.features_size))
        return X_concat, Y_concat
    
    def reshape(self, X, Y):
        X_scaled = np.reshape(X ,(-1, 100, constants.prosody_size))
        Y_scaled = np.reshape(Y ,(-1, 100, constants.features_size  ))
        return X_scaled, Y_scaled
    
    def flatten_x(self, X):
        X_concat = np.reshape(X,(-1,constants.prosody_size))
        return X_concat
    
    def flatten_y(self, Y):
        Y_concat = np.reshape(Y,(-1,constants.features_size ))
        return Y_concat
    
    def reshape_x(self, X):
        X_scaled = np.reshape(X ,(100, constants.prosody_size))
        return X_scaled
    
    def reshape_y(self, Y):
        Y_scaled = np.reshape(Y,(-1, constants.features_size))
        return Y_scaled

    def addDerevative(self, X):
        for i in range(X.shape[2]):
            first = speechpy.processing.derivative_extraction(X[:,:,i], 1)
            second = speechpy.processing.derivative_extraction(first, 1)
            X = np.append(X, first.reshape(X.shape[0], X.shape[1], -1), axis=2)
            X = np.append(X, second.reshape(X.shape[0], X.shape[1], -1), axis=2)
        return X


class TrainSet(Set):

    def __init__(self):
        super(TrainSet, self).__init__("train")

    def scaling(self, flag):
        if flag:
            self.X = self.X_scaled
            self.Y = self.Y_scaled

    def scale_x(self, x):
        x_concat = self.flatten_x(x)
        x_scaled_concat = self.x_scaler.transform(x_concat)
        x_scaled = self.reshape_x(x_scaled_concat)
        return x_scaled

    def scale_y(self, y):
        y_concat = self.flatten_y(y)
        y_scaled_concat = self.y_scaler.transform(y_concat)
        y_scaled = self.reshape_y(y_scaled_concat)
        return y_scaled

    def rescale_y(self, y): 
        #y_concat = self.flatten_y(y)      
        #y_inverse = self.y_scaler.inverse_transform(y_concat)
        #y_scaled = self.reshape_y(y_inverse)
        
        y_inverse = self.y_scaler.inverse_transform(y)
        return y_inverse

class TestSet(Set):

    def __init__(self):
        super(TestSet, self).__init__("test")

    def scaling(self, x_scaler, y_scaler):
        X_concat, Y_concat = self.flatten(self.X, self.Y)
        X_scaled_concat = x_scaler.transform(X_concat)
        Y_scaled_concat = y_scaler.transform(Y_concat)
        
        

        X_scaled, Y_scaled = self.reshape(X_scaled_concat, Y_scaled_concat)


        self.X =  X_scaled
        self.Y = Y_scaled

