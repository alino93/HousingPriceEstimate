from __future__ import print_function
import imageio
from PIL import Image
import numpy as np
import keras
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import Model, load_model

#from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator



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


def show_image(file_add, pred_label, prediction):
    conf = "{:.3f}".format(np.max(prediction))
    title = 'Prediction: '+pred_label[:-4]+'   Confidence: ' + conf

    original = load_img(file_add)
    plt.figure(figsize=[7, 7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()

if __name__ == "__main__":

    num_classes = 7
    class_labels = ['bathroom.jpg','bedroom.jpg','dining_room.jpg','front.jpg','kitchen.jpg','living_room.jpg','satellite.jpg']

    # load model
    model = load_model('room_model.h5')
    model.summary()
    f = open('room_model_history.pckl', 'rb')
    model_history = pickle.load(f)
    f.close()

    # visualize model results
    visualize_results(model_history)

    # preprocess image
    file_add = './pics/room/train/7/151.jpg'
    #img = load_img(file_add, target_size=(224, 224))
    #img = img_to_array(img)
    #img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    #img = preprocess_input(img)
    img = imageio.imread(file_add, pilmode='RGB')
    img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    datagen = ImageDataGenerator(rescale=1. / 255)

    set = datagen.flow(img, batch_size=1)

    # predict
    prediction = model.predict_generator(set, steps=1)
    pred_label = class_labels[int(np.argmax(prediction))]
    print(pred_label)

    # plot 1 of predictions
    show_image(file_add, pred_label, prediction)


    """
    batch data predict
    
    # files direction
    train_dir = './pics/room/train'
    validation_dir = './pics/room/valid'
        
    # Load the normalized images
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    
    # Data generate for train data
    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=300,
        class_mode='categorical')

    # Data generate for validate data
    valid_set = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=300,
        class_mode='categorical',
        shuffle=False)
    
    # predict on all batches for train set
    predictions = model.predict_generator(train_set, steps=train_set.samples / train_set.batch_size, verbose=1)
    # predict on all batches for valid set
    predictions2 = model.predict_generator(valid_set,steps=valid_set.samples / valid_set.batch_size, verbose=1)
    
    #save predictions
    f = open('room_pred.pckl', 'wb')
    pickle.dump(predictions2, f)
    f.close()
    
    """
    ## restore labels for all images and categories FC8[cat,house,feature]
    #f = open('FC8.pckl', 'rb')
    #FC8 = pickle.load(f)
    #f.close()

