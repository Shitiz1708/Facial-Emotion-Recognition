# from keras.preprocessing.image import load_img,img_to_array
import os
import math
import numpy as np
import cv2

X_train=[]
y_train = []

emotion_folder = os.listdir('./Emotion')
for i in emotion_folder:
    # print(i)
    inner_folder = os.listdir('./Emotion/'+i+'/')
    print(inner_folder)
    for j in inner_folder:
        # print(j)/
        print('./Emotion/' + str(i) +'/' + str(j) + '/')
        txtfile = os.listdir('./Emotion/' + i +'/' + j + '/')
        print(txtfile)
        if len(txtfile)!=0:
            with open('./Emotion/' + i +'/' + j + '/'+txtfile[0],'r') as file:
                label = int(file.read().split(' ')[3][0])
                print(label)
                y_train.append(label)
            path = './cohn-kanade-images/' + i + '/' + j + '/'
            images1 = sorted(os.listdir(path),key=len)
            if '.DS_Store' in images1:
                images1.remove('.DS_Store')
            
            images = []
            # print(len(images1))
            if len(images1)>6:
                images = sorted(np.random.choice(images1,6))
                # for k in range(0,len(images1),int(len(images1)/7)):
                #     images.append(images1[k])
            else:
                images = images1
            print(len(images))
            print(images)
            imgs = []
            for image in images:
                # print(image)
                im = cv2.imread(path+ image)
                im = cv2.resize(im,(640,480),interpolation = cv2.INTER_AREA)
                # im = load_img(path + image)
                # im = img_to_array(im,data_format="channels_first")
                imgs.append(im)
                print(im.shape)
            imgs = np.array(imgs)
            print(imgs.shape)
            X_train.append(imgs)

X_train = np.array(X_train)
print(X_train.shape)
y_train = np.array(y_train)
print(y_train.shape)
np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)

                
