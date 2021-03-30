import numpy as np
import netCDF4 as nc
import scipy.io as sio
file=sio.loadmat('ERA_grid.mat')
lat=file['lat']
lon=file['lon']

fileList_test=[]
fileList_test.append('geopotential_500hPa_2018_5.625deg.nc')

file=nc.Dataset('ERA_Z500_1hour.nc')
Z500=np.asarray(file['input'])
M=np.mean(Z500.flatten())
sdev=np.std(Z500.flatten())

from matplotlib import pyplot as plt
F=nc.Dataset(fileList_test[0])
Z=np.asarray(F['z'])
TRUTH=Z
[qx,qy]=np.meshgrid(lon,lat)

Z_rs = np.reshape(Z,[np.size(Z,0), int(np.size(Z,1)*np.size(Z,2))])
TRUTH = Z_rs
Z_rs = (Z_rs-M)/sdev
TRUTH = (TRUTH-M)/sdev
noise=3
for k in range(1,np.size(Z_rs,0)):
 Z_rs[k-1,:]=Z_rs[k-1,:]+np.random.normal(0, noise, 2048)
 

print('length of initial condition',len(Z_rs[0,:]))

def ENKF(x, n, P ,Q, R, obs, model, u_ensemble):
    obs=np.reshape(obs,[n,1]) 
    x=np.reshape(x,[n,1])
    [U,S,V]=np.linalg.svd(P)
    D=np.zeros([n,n])
    np.fill_diagonal(D,S)
    sqrtP=np.dot(np.dot(U,np.sqrt(D)),U)
    ens=np.zeros([n,2*n])
    ens[:,0:n]=np.tile(x,(1,n)) + sqrtP
    ens[:,n:]=np.tile(x,(1,n)) - sqrtP
    ## forecasting step,dummy model

    for k in range(0, np.size(ens,1)):

       u =  model.predict(np.reshape(ens[:,k],[1, 32, 64, 1]))

       u_ensemble[:,k]=np.reshape(u,(32*64,))



    ############################
    x_prior = np.reshape(np.mean(u_ensemble,1),[n,1])
    print('shape pf x_prior',np.shape(x_prior))
    print('shape pf obs',np.shape(obs))
    cf_ens = ens - np.tile(x_prior,(1,2*n))
    P_prior = np.dot(cf_ens,np.transpose(cf_ens))/(2*n - 1)+Q
    h_ens = ens
    y_prior=np.reshape(np.mean(h_ens,1),[n,1])
    ch_ens = h_ens - np.tile(y_prior,(1,2*n))
    print('shape pf y_prior',np.shape(y_prior))
    P_y = np.dot(ch_ens, np.transpose(ch_ens))/(2*n-1) + R
    P_xy = np.dot(cf_ens, np.transpose(ch_ens)) /(2*n-1)
    K = np.dot(P_xy,np.linalg.inv(P_y))
    P = P_prior - np.dot(np.dot(K,P_y),np.transpose(K))
    x = x_prior + np.dot(K,(obs-y_prior))

    return x, P






import tensorflow
import keras.backend as K
#from data_manager import ClutteredMNIST
#from visualizer import plot_mnist_sample
#from visualizer import print_evaluation
#from visualizer import plot_mnist_grid
import netCDF4
import numpy as np
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
import keras
from keras.callbacks import History
history = History()

import keras
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D
from keras.models import load_model

__version__ = 0.1

def CConv2D(filters, kernel_size, strides=(1, 1), activation='linear', padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None):
    def CConv2D_inner(x):
        # padding (see https://www.tensorflow.org/api_guides/python/nn#Convolution)
        in_height = int(x.get_shape()[1])
        in_width = int(x.get_shape()[2])

        if (in_height % strides[0] == 0):
            pad_along_height = max(kernel_size[0] - strides[0], 0)
        else:
            pad_along_height = max(
                kernel_size[0] - (in_height % strides[0]), 0)
        if (in_width % strides[1] == 0):
            pad_along_width = max(kernel_size[1] - strides[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (in_width % strides[1]), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # left and right side for padding
        pad_left = Cropping2D(cropping=((0, 0), (in_width-pad_left, 0)))(x)
        pad_right = Cropping2D(cropping=((0, 0), (0, in_width-pad_right)))(x)

        # add padding to incoming image
        conc = Concatenate(axis=2)([pad_left, x, pad_right])

        # top/bottom padding options
        if padding == 'same':
            conc = ZeroPadding2D(padding={'top_pad': pad_top,
                                          'bottom_pad': pad_bottom})(conc)
        elif padding == 'valid':
            pass
        else:
            raise Exception('Padding "{}" does not exist!'.format(padding))

        # perform the circular convolution
        cconv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, activation=activation,
                         padding='valid',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(conc)

        # return circular convolution layer
        return cconv2d
    return CConv2D_inner

from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

from utils import get_initial_weights
from layers import BilinearInterpolation


def stn(input_shape=(32, 64, 1), sampling_size=(8, 16), num_classes=10):
    image = Input(shape=input_shape)
    #locnet = Conv2D(32, (5, 5), padding='same')(image)
    locnet = CConv2D(32, (5, 5), padding='same')(image)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    #locnet = Conv2D(32, (5, 5), padding='same')(locnet)
    locnet = CConv2D(32, (5, 5), padding='same')(locnet)

    locnet = Activation('relu')(locnet)
    locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    #locnet = CConv2D(32, (5, 5), padding='same')(locnet)

    #locnet = Conv2D(20, (5, 5), padding='same')(locnet)
    #locnet = Activation('relu')(locnet)
    #locnet = MaxPool2D(pool_size=(2, 2))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(500)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(200)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(100)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([image, locnet])
    #x = Conv2D(32, (3, 3), padding='same')(x)
    x = CConv2D(32, (5, 5), padding='same')(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    #x=  Conv2D(32, (3,3), padding='same')(x)
    x = CConv2D(32, (5, 5), padding='same')(x)

    x = Activation('relu')(x)
    x = UpSampling2D (size=(2,2))(x)
    #x = Conv2D(32, (3,3), padding='same')(x)
    #x = CConv2D(32, (5, 5), padding='same')(x)

    #x = Activation('relu')(x)
    #x = Conv2D(32, (3,3), padding='same')(x)
    #x = CConv2D(32, (5, 5), padding='same')(x)

    #x = Activation('relu')(x)
    #x = UpSampling2D (size=(2,2))(x)
    #x = Conv2D(2, (3,3), padding='same')(x)
    x = CConv2D(1, (5, 5), padding='same')(x)

    x = Activation('linear')(x)
    return Model(inputs=image, outputs=x)

model = stn()
model.load_weights('best_weights_lead1.h5')
###### Start Data Assimilation Process #########################################

time = 1200
n=int(32*64)
P=np.eye(n,n)

Q=0.03*np.eye(n,n)

R=0.0001

u_ensemble=np.zeros([32*64,2*32*64])

pred=np.zeros([time,32,64,1])


dt=24
count=0
for t in range(0, time, dt):
    
    for kk in range(0,dt-1):
        if (kk==0):   
          u=Z_rs[t+kk,:].reshape([1, 32, 64, 1 ])
          u=model.predict(u.reshape([1,32,64,1]))
        else :
      
          u=model.predict(u)
        
        pred[count,:,:,0]=np.reshape(u,[32,64])
        count=count+1
    x=u   
    x, P = ENKF(x, 2048, P, Q, R, Z_rs[t+dt,:], model,u_ensemble)
   
    print('output shape of ENKF', np.shape(x))
    
    pred[count,:,:,0]=np.reshape(x,[32,64])
    count=count+1


sio.savemat('DA_every24HR_lead1200_everytime_noise_' + str(noise)+ '.mat',dict([('prediction',pred),('truth',np.reshape(TRUTH,[np.size(Z_rs,0),32,64,1])),('noisy_obs',np.reshape(Z_rs,[np.size(Z_rs,0),32,64,1]))]))

print('Done writing file')
