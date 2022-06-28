import keras
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.callbacks import LearningRateScheduler

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

# updatable plot
# a minimal example (sort of)

class PlotLosses(keras.callbacks.Callback):
    '''
    Lossのグラフプロット用
    '''
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.title("Learning progress")
        plt.show();

class PlotLossesIterativeFit(keras.callbacks.Callback):
    '''
    Lossのグラフプロット用. 何度もfitが呼ばれる場合用
    '''
    def __init__(self,num_iteration):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []        
        
        self.num_iteration = num_iteration

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.title("Learning progress: total " + str(self.num_iteration)+" iterations")
        plt.show();

class PlotLossesTrainOnBatch():
    '''
    Lossのグラフプロット用. train on batch用。train on batchが終わるとadd_lossを呼び出す必要がある
    '''
    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        
        self.fig = plt.figure()
        

    def add_loss(self,loss,plot=True):
        self.x.append(self.i)
        self.i += 1
        self.losses.append(loss)
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.title("Learning progress")
        plt.show();

def split_params(W, U, b, latent_dim):
    '''
    LSTMのパラメータをばらす
    '''
    Wi = W[:,0:latent_dim]
    Wf = W[:,latent_dim:2*latent_dim]
    Wc = W[:,2*latent_dim:3*latent_dim]
    Wo = W[:,3*latent_dim:]

    #print("Wi : ",Wi.shape)
    #print("Wf : ",Wf.shape)
    #print("Wc : ",Wc.shape)
    #print("Wo : ",Wo.shape)

    Ui = U[:,0:latent_dim]
    Uf = U[:,latent_dim:2*latent_dim]
    Uc = U[:,2*latent_dim:3*latent_dim]
    Uo = U[:,3*latent_dim:]

    #print("Ui : ",Ui.shape)
    #print("Uf : ",Uf.shape)
    #print("Uc : ",Uc.shape)
    #print("Uo : ",Uo.shape)

    bi = b[0:latent_dim]
    bf = b[latent_dim:2*latent_dim]
    bc = b[2*latent_dim:3*latent_dim]
    bo = b[3*latent_dim:]
    #print("bi : ",bi.shape)
    #print("bf : ",bf.shape)
    #print("bc : ",bc.shape)
    #print("bo : ",bo.shape)

    return (Wi, Wf, Wc, Wo), (Ui, Uf, Uc, Uo), (bi, bf, bc, bo)


def calc_ht(params):
    '''
    LSTMの出力を計算
    '''
    x, latent_dim, W_, U_, b_ = params
    Wi, Wf, Wc, Wo = W_
    Ui, Uf, Uc, Uo = U_
    bi, bf, bc, bo = b_ 
    n = x.shape[0]

    ht_1 = np.zeros(n*latent_dim).reshape(n,latent_dim) #h_{t-1}を意味する．
    Ct_1 = np.zeros(n*latent_dim).reshape(n,latent_dim) #C_{t-1}を意味する．

    ht_list = []

    for t in np.arange(x.shape[1]):
        xt = np.array(x[:,t,:])
        it = sigmoid(np.dot(xt, Wi) + np.dot(ht_1, Ui) + bi)
        Ct_tilda = np.tanh(np.dot(xt, Wc) + np.dot(ht_1, Uc) + bc)
        ft = sigmoid(np.dot(xt, Wf) + np.dot(ht_1, Uf) + bf)
        Ct = it * Ct_tilda + ft * Ct_1
        ot = sigmoid( np.dot(xt, Wo) + np.dot(ht_1, Uo) + bo)
        ht = ot * np.tanh(Ct)
        ht_list.append(ht)
        ht_1 = ht
        Ct_1 = Ct

    ht = np.array(ht)
    return ht

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
        

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    return LearningRateScheduler(schedule)

#how to use
#lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
#model.fit(X_train, Y_train, callbacks=[lr_sched])    
