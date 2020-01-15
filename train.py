import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,GlobalAveragePooling2D,TimeDistributed,ConvLSTM2D,MaxPooling3D,Flatten,BatchNormalization,LSTM,Dropout
from keras.models import Model
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.datasets import mnist
from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import cv2

X_train = np.load('X_train_resnet_video.npy')
print(X_train.shape)
y_train = np.load('y_train_resnet_video.npy')

y_train = to_categorical(y_train,8)

input_tensor = Input(shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3],X_train.shape[4]))
x = TimeDistributed(Flatten())(input_tensor)
x = LSTM(256,dropout=0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
prediction = Dense(8, activation='softmax')(x)
model = Model(inputs = input_tensor,outputs = prediction)
sgd = SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
model.fit(X_train[:100],y_train[:100], batch_size=2,epochs=30, validation_split=0.05,callbacks=callbacks)
print(model.summary())
