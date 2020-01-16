import numpy as np
import keras
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import load_model
import cv2
import os

frames = []
path = './test/'
imgs = sorted(os.listdir(path))
if len(imgs)>6:
    imgs = imgs[-6:]

base_model = ResNet50(include_top = False,weights = 'imagenet')
for im in imgs:
    im = cv2.imread(path+ im)
    im = cv2.resize(im,(224,224),interpolation = cv2.INTER_AREA)
    im = np.expand_dims(im,axis=0)
    im = preprocess_input(im)
    feature = base_model.predict(im)
    frames.append(feature[0])

frames = np.array(frames)
frames = np.expand_dims(frames,axis=0)

print(frames.shape)

model = load_model('video_1_LSTM_1_1024.h5')
print(model.predict(frames))
