import numpy as np
import netCDF4 as nc
import scipy.io as sio

file=sio.loadmat('/global/cscratch1/sd/ashesh/MyQuota/ERA5/Data_assimilation/ERA_grid.mat')
lat=file['lat']
lon=file['lon']

file=nc.Dataset('/global/cscratch1/sd/ashesh/MyQuota/ERA5/ERA_Z500_1hour.nc')
Z=np.asarray(file['input'])
M=np.mean(Z.flatten())
sdev=np.std(Z.flatten())

fileList_train = []
fileList_test=[]
for i in range (1979,2017):
    fileList_train.append ('/global/cscratch1/sd/ashesh/MyQuota/ERA5/geopotential_500hPa_' + str(i)+'_5.625deg.nc')

fileList_test.append('/global/cscratch1/sd/ashesh/MyQuota/ERA5/geopotential_500hPa_2018_5.625deg.nc')


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
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, Concatenate, ZeroPadding2D, merge
from keras.models import load_model
from keras.layers import Input
from keras.models import Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense

from utils import get_initial_weights
from layers import BilinearInterpolation

__version__ = 0.1


def unet_baseline(input_shape=(32, 64, 1), sampling_size=(8, 16), num_classes=10):
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv3)


    conv5 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv3)
    x = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv5)



    up6 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, 2, 2,activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(x)), conv2])
    conv6 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv6)

    up7 = keras.layers.Concatenate(axis=-1)([Convolution2D(32, 2, 2,activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv1])
    conv7 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(conv7)

    conv10 = Convolution2D(1, 5, 5, activation='linear',border_mode='same')(conv7)

    model = Model(input=inputs, output=conv10)



    return model

model = unet_baseline()
model.compile(loss='mse', optimizer='adam')
model.summary()


batch_size = 10     ### This has undergone HPO. Don't change
num_epochs = 8      #### This has undergone HPO. But less sensitive to change
lead = 1            #### See paper for details on this variable. This lead refers to "x" in U-NETx
count=0
for loop in fileList_train:
    print('******************** counter*************',count)
    File=nc.Dataset(loop)
    Z=np.asarray(File['z'])
    trainN=np.size(Z,0)-300
    Z=(Z-M)/sdev

    x_train=Z[0:trainN,:,:]
    x_train=x_train.reshape([np.size(x_train,0),32,64,1])
    y_train=Z[lead:trainN+lead,:,:]
    y_train=y_train.reshape([np.size(y_train,0),32,64,1])

    x_val= Z[trainN+lead:np.size(Z,0)-lead,:,:]
    x_val=x_val.reshape([np.size(x_val,0),32,64,1])

    y_val= Z[trainN+lead*2:np.size(Z,0),:,:]
    y_val=y_val.reshape([np.size(y_val,0),32,64,1])

    if (count>0):

        model = stn()
        model.compile(loss='mse', optimizer='adam')
        model.load_weights('best_weights_lead1.h5')
        hist = model.fit(x_train, y_train,
                       batch_size = batch_size,
             verbose=1,
             epochs = 20,
             validation_data=(x_val,y_val),shuffle=True,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights_lead'+str(lead)+'.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )

    else:
        hist = model.fit(x_train, y_train,
                       batch_size = batch_size,
             verbose=1,
             epochs = 20,
             validation_data=(x_val,y_val),shuffle=True,
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights_lead'+str(lead)+'.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1),history]
             )


    count=count+1
### This is the autoregressive prediction with baseline U-NET without DA #### 
model.load_weights('best_weights_lead'+str(lead)+'.h5')
F=nc.Dataset(fileList_test[0])
Z=F['z']
Z=(Z-M)/sdev
testN=1000
x_test=Z[0:testN,:,:]
x_test=x_test.reshape([np.size(x_test,0),32,64,1])
y_test=Z[lead:testN+lead,:,:]
y_test=y_test.reshape([np.size(y_test,0),32,64,1])

pred=np.zeros([1000,32,64,1])
for k in range (0, 1000):
    if(k==0):
      pred[k,:,:,0]=model.predict(x_test[k,:,:,0].reshape([1,32,64,1])).reshape([32,64])
    else:
      pred[k,:,:,0]=model.predict(pred[k-1,:,:,0].reshape([1,32,64,1])).reshape([32,64])

sio.savemat('ERA5_1_hr_UNet_noSTN.mat',dict([('prediction',pred),('truth',y_test)]))

print('Finished writing File')
