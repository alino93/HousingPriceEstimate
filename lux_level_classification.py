import os
import numpy as np
import pandas as pd

import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from pathlib import Path, PureWindowsPath
from sklearn.model_selection import GridSearchCV
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list

def predict_knn(feature_train, label_train, k):
    # find nearest neighbors
    neigh = KNeighborsClassifier(n_neighbors=k).fit(feature_train, label_train)

    return neigh


def classify_knn(feature_train, label_train, feature_test, label_test, n):
    # predict label KNN
    neigh = predict_knn(feature_train, label_train, 2)

    label_test_pred = neigh.predict(feature_test)
    label_test_pred = label_test_pred.astype('int64')

    label_classes = ['1', '2', '3', '4', '5', '6', '7', '8']
    confusion, accuracy = compute_confusion_matrix(label_test_pred, label_test, n)

   # visualize_confusion_matrix(confusion, accuracy, label_classes[:n], 'KNN Confusion Matrix')

    return label_test_pred

def classify_svm(x_train, y_train, x_test, n):
    # train label SVM
    #t0 = time()
    #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    #clf = GridSearchCV(
    #    SVC(kernel='rbf', class_weight='balanced'), param_grid
    #)
    #clf = clf.fit(x_train, y_train)
    #print("Best estimator found by grid search:")
    #print(clf.best_estimator_)

    clf = SVC(kernel='rbf', class_weight='balanced', C=1e3, gamma=0.1)
    clf = clf.fit(x_train, y_train)

    # predict using svm
    y_pred = clf.predict(x_test)

    return y_pred

def classify_cnn(model, categ, num_classes):
    data_dir = './pics' + categ + '/valid'
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



if __name__ == '__main__':

    # read meta data
    excel = pd.read_excel('text_datas.xlsx')
    excel = excel.drop('Unnamed: 0', axis=1)

    # number of data
    num = excel.shape[0]

    # number of price levels
    num_classes = 5

    ## restore true labels for all images and categories FC8[cat,house,feature] from googleNet
    f = open('FC8_s.pckl', 'rb')
    FC8 = pickle.load(f)
    categories = ['bathroom.jpg', 'bedroom.jpg', 'dining_room.jpg', 'front.jpg', 'kitchen.jpg', 'living_room.jpg',
                  'satellite.jpg']
    f.close()

    level = excel.iloc[:, 4].values
    level -= min(level)
    level = level/(750000+1)
    level = np.floor(level*num_classes) + 1
    level[level > num_classes] = num_classes


    #level = np.floor(np.arange(0,num,1) / num * num_classes) + 1


    unique, counts = np.unique(level, return_counts=True)
    dict(zip(unique, counts))

    label_knn_pred = np.zeros((7,num))
    label_knn = level
    label_svm_pred = np.zeros((7, num))
    label_svm = label_knn

    for j in range(len(categories)):
        X = FC8[j]
        Y = level

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        #classify using knn
        label_knn_pred[j] = classify_knn(X_train, Y_train, X, Y, num_classes)

        #classify using svm
        label_svm_pred[j] = classify_svm(X_train, Y_train, X, num_classes)

        #classify using cnn
        # run vgg_lux_level_classification.py to train cnn

    label_pred_mode_knn, counts = mode(label_knn_pred, axis=0)
    label_pred_mode_svm, counts2 = mode(label_svm_pred, axis=0)

    confusion_knn, accuracy_knn = compute_confusion_matrix(label_pred_mode_knn.reshape(-1), label_knn, num_classes)
    confusion_svm, accuracy_svm = compute_confusion_matrix(label_pred_mode_svm.reshape(-1), label_svm, num_classes)

    label_classes = ['1', '2', '3', '4', '5', '6', '7', '8']

    visualize_confusion_matrix(confusion_knn, accuracy_knn, label_classes[:num_classes], 'KNN Confusion Matrix')
    visualize_confusion_matrix(confusion_svm, accuracy_svm, label_classes[:num_classes], 'SVM Confusion Matrix')

    #f = open('level_knn.pckl', 'wb')
    #pickle.dump(label_pred_mode_knn.reshape(-1), f)
    #f.close()

    #f = open('level_svm.pckl', 'wb')
    #pickle.dump(label_pred_mode_svm.reshape(-1), f)
    #f.close()



