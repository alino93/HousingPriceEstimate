# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sWvpIIIIWClMTq5CwjIUX_HeITiFMm-c
"""

from __future__ import print_function
import imageio
from PIL import Image
import numpy as np
import keras
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import mode
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from keras.models import Model, load_model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16
from tensorflow.python.client import device_lib
from keras.utils.vis_utils import plot_model

#!unzip pics.zip
# Utility function for plotting of the model results

def fit_vgg(train_batchsize, val_batchsize, categ ,num_classes):

    # files direction
    train_dir = './' + categ + '/train'
    validation_dir = './' + categ + '/valid'

    # Init the VGG model
    vgg = vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
    # Freeze all the layers
    for layer in vgg.layers[:-3]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg.layers:
        print(layer, layer.trainable)

    # Create the model
    x = Flatten()(vgg.output)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # create graph of your new model
    model = Model(inputs=vgg.input, outputs=predictions)

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()
    plot_model(model, to_file='vgg.png')
    
    # compile the model
    model.compile(optimizer=RMSprop(lr=1e-4, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

    # Load the rescaled images
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    #train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,
    #    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Data generate for train data
    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical',
        shuffle=False)

    # Data generate for validate data
    valid_set = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)
    # Train the model
    model_history = model.fit_generator(
        train_set,
        steps_per_epoch=train_set.samples / train_set.batch_size,
        epochs=15,
        validation_data=valid_set,
        validation_steps=valid_set.samples / valid_set.batch_size,
        verbose=1)

    return model_history, model

def classify_cnn(model, categ, num_classes):
    data_dir = './' + categ + '/data'
    data_datagen = ImageDataGenerator(rescale=1. / 255)
    data_set = data_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=100,
        class_mode='categorical',
        shuffle=False)
    predictions = model.predict_generator(data_set, steps=data_set.samples / data_set.batch_size, verbose=1)
    label_pred = np.argmax(predictions,axis=1)+1
    true_label = data_set.classes + 1
    return label_pred, true_label  

def compute_confusion_matrix(label_pred, label_test, n):
    label_test_pred = label_pred.astype('int64')
    label_test = label_test.astype('int64')
    num_test = label_test.shape[0]
    acc = 0
    confusion = np.zeros((n, n))
    for i in range(num_test):
        confusion[label_test_pred[i] - 1, label_test[i] - 1] += 1
        if label_test_pred[i] == label_test[i]:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(n):
        if np.sum(confusion[:, i]) != 0:
            confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])
    return confusion, accuracy

def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def visualize_confusion_matrix(confusion, accuracy, label_classes, name):
    plt.title("{}, accuracy = {:.3f}".format(name, accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()

if __name__ == "__main__":
    #drive.mount('/content/gdrive')
    
    # read meta data
    excel = pd.read_excel('text_datas.xlsx')
        
    # number of data
    num = excel.shape[0]-3
    num_classes = 5

    train_batchsize = 250
    val_batchsize = 50
    categories = ['bathroom.jpg', 'bedroom.jpg', 'dining_room.jpg', 'front.jpg', 
                  'kitchen.jpg', 'living_room.jpg', 'satellite.jpg']
    
    label_cnn_pred = np.zeros((7,481))
    for j in range(len(categories)):
        # category
        categ = categories[j][:-4]

        model_history, model = fit_vgg(train_batchsize, val_batchsize, categ, num_classes)
        # save model
        model.save(categ+'_model.h5')
        print("Saved model to disk")
        
        #classify using cnn
        label_cnn_pred[j], label_cnn = classify_cnn(model, categ, num_classes)

        visualize_results(model_history)
    
    label_pred_mode_cnn, counts = mode(label_cnn_pred, axis=0)

    confusion_cnn, accuracy_cnn = compute_confusion_matrix(label_pred_mode_cnn.reshape(-1), label_cnn, num_classes)

    label_classes = ['1', '2', '3', '4', '5', '6', '7', '8']

    visualize_confusion_matrix(confusion_cnn, accuracy_cnn, label_classes[:num_classes], 'CNN Confusion Matrix')

    f = open('level_cnn.pckl', 'wb')
    pickle.dump(label_pred_mode_cnn, f)
    f.close()
