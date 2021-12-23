from __future__ import print_function
import imageio
from PIL import Image
import numpy as np
import pickle
import os, shutil, random
import pandas as pd
from os import path

if __name__ == "__main__":

    num_classes = 5

    # read sorted meta data (sorted based on price)
    excel = pd.read_excel('text_datas.xlsx')

    # number of data
    num = excel.shape[0]
    excel = excel.drop('Unnamed: 0', axis=1)

    # lux levels
    level = excel.iloc[:, 4].values
    level -= min(level)
    level = level / (750000 + 1)
    level = np.floor(level * num_classes) + 1
    level[level > num_classes] = num_classes

    categories = ['bathroom.jpg','bedroom.jpg','dining_room.jpg','front.jpg','kitchen.jpg','living_room.jpg','satellite.jpg']

    # copy files to directories based on price level
    for j in range(len(categories)):
        for i in range(num):
            # load img of each house
            add = excel.at[i, 'Address']
            file_add = 'pics\\'+add+'\\'+categories[j]
            try:
                img = imageio.imread(file_add, pilmode='RGB')
            except IOError:
                continue
            folderPath = 'pics\\' + categories[j][:-4] + '\\' + 'train' + '\\' + str(int(level[i])) #str(int(np.floor(i/num*num_classes))+1)
            folderPath2 = 'pics\\' + categories[j][:-4] + '\\' + 'data' + '\\' + str(int(level[i]))

            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
                os.makedirs(folderPath2)
                shutil.copy(file_add, os.path.join(folderPath,str(i)+'.jpg'))
                shutil.copy(file_add, os.path.join(folderPath2, str(i) + '.jpg'))
            else:
                shutil.copy(file_add, os.path.join(folderPath,str(i)+'.jpg'))
                shutil.copy(file_add, os.path.join(folderPath2, str(i) + '.jpg'))


    # randomly move 20% of files to valid folder
    for j in range(len(categories)):
        for i in range(num_classes):
            for k in range(int(num//num_classes*0.21)):
                folderPath = 'pics\\' + categories[j][:-4] + '\\' + 'train' + '\\' + str(int(i)+1)
                file = random.choice(os.listdir(folderPath))
                desPath = 'pics\\' +categories[j][:-4] + '\\' + 'valid' + '\\' + str(int(i)+1)
                if not os.path.exists(desPath):
                    os.makedirs(desPath)
                    shutil.move(folderPath + '\\' + file, desPath)
                else:
                    shutil.move(folderPath + '\\' + file, desPath)


    ## restore labels for all images and categories FC8[cat,house,feature]
    #f = open('FC8.pckl', 'rb')
    #FC8 = pickle.load(f)
    #f.close()

    #4 best labels from best to worst
    #ssss = a.argsort()[-4:][::-1]
    #predicted_class_name = labels[ssss]
    #print('Predicted Class: ', ssss, ', Class Name: ', predicted_class_name)


    # out[2] is the softmax prediction from the actual classifier (primary classifier of our interest)
    # whereas out[0] and out[1] are outputs of auxiliary classifiers just used to make sure that the model converges faster during training.
    # There is little to no use of the auxiliary classifiers at inference time.
