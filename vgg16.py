from __future__ import print_function
import imageio
from PIL import Image
import numpy as np
import keras
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, Concatenate, Reshape, Activation
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pool_helper import PoolHelper
from lrn import LRN
from keras.layers.normalization import BatchNormalization
from keras.applications import vgg16
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Utility function for plotting of the model results
def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
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


def fit_vgg(train_batchsize, val_batchsize, num_classes):

    # files direction
    train_dir = './pics/room/train'
    validation_dir = './pics/room/valid'

    # Init the VGG model
    vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(3, 224, 224))
    # Freeze all the layers
    for layer in vgg.layers[:-3]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg.layers:
        print(layer, layer.trainable)

    x = Flatten()(vgg.output)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # create graph of your new model
    model = Model(input=vgg.input, output=predictions)

    # compile the model
    model.compile(optimizer=RMSprop(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Load the rescaled images
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Data generate for train data
    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=train_batchsize,
        class_mode='categorical')

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
        epochs=20,
        validation_data=valid_set,
        validation_steps=valid_set.samples / valid_set.batch_size,
        verbose=1)

    # evaluate model
    # scores = model.evaluate(X, Y, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    return model_history, model

if __name__ == "__main__":

    num_classes = 7

    # files direction
    train_dir = './pics/room/train'
    validation_dir = './pics/room/valid'

    train_batchsize = 137
    val_batchsize = 36

    model_history, model = fit_vgg(train_batchsize, val_batchsize, num_classes)

    visualize_results(model_history)

    # save model
    model.save("room_model2.h5")
    print("Saved model to disk")
    f = open('room_model_history2.pckl', 'wb')
    pickle.dump(model_history, f)
    f.close()

    # load model
    #model = load_model('bathroom_model.h5')
    #model.summary()
    #f = open('room_model_history.pckl', 'rb')
    #model_history = pickle.load(f)
    #f.close()
    #visualize_results(model_history)

    # predict
    #predictions = model.predict_generator(train_set, steps=train_set.samples / train_set.batch_size, verbose=1)

    """
    FC8 = np.zeros((7,177,1000))
    categories = ['bathroom.jpg','bedroom.jpg','dining_room.jpg','front.jpg','kitchen.jpg','living_room.jpg','satellite.jpg']
    for j in range(len(categories)):
        for i in range(177):
            # load front img of each house
            add = excel.at[i, 'Address']
            file_add = 'pics\\'+add+'\\'+categories[j]
            try:
                img = imageio.imread(file_add, pilmode='RGB')
            except IOError:
                continue

            img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
            img[:, :, 0] -= 123.68
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 103.939
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)

            # get googlenet output
            out = model.predict(img) # note: the model has three outputs
            #predicted_label = np.argmax(out[2])
            #predicted_class_name = labels[predicted_label]
            #print('Predicted Class: ', predicted_label, ', Class Name: ', predicted_class_name)
            # store all labels FC8
            FC8[j][i] = np.reshape(out[2], (1000,))
            f = open('FC8.pckl', 'wb')
            pickle.dump(FC8, f)
            f.close()

    level = excel.iloc[:, 4].values
    level -= min(level)
    level = level/(max(level)+1)
    level = np.floor(level*8) + 1
    """
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
