from vit_keras import vit, utils
from keras.applications.densenet import DenseNet201, preprocess_input
from keras.applications.convnext import ConvNeXtXLarge, preprocess_input
import numpy as np
from keras.preprocessing import image
from numpy import linalg as LA

import os
from tkinter import Label, Menu, filedialog
import SimpleITK as sitk
import csv
import six
import time

image_size = 384
classes = utils.get_imagenet_classes()
model = vit.vit_l32( #vit_b16, vit_b32
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    weights='imagenet21k+imagenet2012'
)
modelCNN = DenseNet201(weights='imagenet', input_shape = (224, 224, 3), pooling = 'max', include_top = False)
modelCon = ConvNeXtXLarge(weights='imagenet', input_shape = (224, 224, 3), pooling = 'max', include_top = False)

csvname="l32Dense_Features_Buld.csv"

def read_directory(folder_path,directory_path, dirname):
    #篩選副檔名為jpg檔案
    if directory_path[-3:]=='jpg':
        global count
        global foldername

        imagePath=directory_path
       
        #為了寫入csv建立list featurename 和featurevalue
        featurename=list()
        featurevalue=list()
        #read image
        #image = sitk.ReadImage(imagePath)
        #開啟並準備寫入csv
        with open(csvname, 'a+',newline='',encoding='utf-8') as data_file:
            writer = csv.writer(data_file)
            #先增加標頭filename
            featurename.append("filename")
            #讀取image檔名
            filename=os.path.splitext(os.path.basename(imagePath))[0]
            #將檔名加入featurevalue list中
            featurevalue.append(filename)
            featurevalue.append(foldername)

            #ViT特徵抓取
            imageLoad = utils.read(imagePath, image_size)
            X = vit.preprocess_inputs(imageLoad).reshape(1, image_size, image_size, 3)
            #start_time = time.time()
            y = model.predict(X)
            #endtime = time.time() - start_time
            norm_feat = y[0]/LA.norm(y[0])
            
            #CNN
            img = image.load_img(imagePath, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = modelCNN.predict(img)
            norm_featCNN = feat[0]/LA.norm(feat[0])
            '''
            #Con
            img = image.load_img(imagePath, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = modelCon.predict(img)
            norm_featCon = feat[0]/LA.norm(feat[0])
            '''
            
            for val in norm_feat:
            #    print(val)
                #將特徵名稱與特徵值放進list中
                featurevalue.append(val)
            
            for val in norm_featCNN:
            #    print(val)
                #將特徵名稱與特徵值放進list中
                featurevalue.append(val)
            
            writer.writerow(featurevalue)

folder_path = filedialog.askdirectory()
folder_content = os.listdir(folder_path)

def show_folder_content(folder_path):
    print(folder_path + '資料夾內容：')
    global foldername
    folder_content = os.listdir(folder_path)
    for item in folder_content:
        if os.path.isdir(folder_path + '\\' + item):
            print('資料夾：' + item)
            foldername=item

            # 呼叫自己處理這個子資料夾
            show_folder_content(folder_path + '\\' + item)
        elif os.path.isfile(folder_path + '\\' + item):
            print('檔案：' + item)
            imagepath=folder_path + '\\' + item
            imagename=item
        
            read_directory(folder_path,imagepath, imagename)
        else:
            print('無法辨識：' + item)

show_folder_content(folder_path)

#url = 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg'
#image = utils.read(url, image_size)
#X = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
#y = model.predict(X)
#norm_feat = y[0]/LA.norm(y[0])
#print(y[0])
#print(classes[y[0].argmax()]) # Granny smith

#fine-tune
'''
image_size = 224
model = vit.vit_l32(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=200
)'''