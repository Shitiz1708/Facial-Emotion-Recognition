from keras.preprocessing.image import load_img,img_to_array
import os
import math
import numpy as np

X_train=[]
y_train = []

emotion_folder = os.listdir('/Emotion')
for i in emotion_folder:
    inner_folder = os.listdir('/Emotion/'+i+'/')
    for j in inner_folder:
        txtfile = os.listdir('/Emotion/' + i +'/' + j + '/')
        if len(txtfile)!=0:
            with open(txtfile[0],'r') as file:
                label = int(file.read().split(' ')[3][0])
                y_train.append(label)
            path = '/cohn-kanade-images/' + i + '/' + j + '/'
            images1 = sorted(os.listdir(path),key=len)
            images = []
            if len(images1)>7:
                # images = sorted(np.random.choice(images,7))
                for i in range(0,len(images1),len(images1)/7):
                    images.append(images1[i])
            else:
                images = images1
            imgs = []
            for image in images:
                im = load_img(path + image)
                im = img_to_array(im,data_format="channels_first")
                imgs.append(im)
            X_train.append(imgs)

X_train = np.array(X_train)
print(X_train.shape)
y_train = np.array(y_train)
print(y_train.shape)
np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)

                
