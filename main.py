

if __name__ == "__main__":

    # Room data process
    exec(open('data_process_room.py').read())

    # Train room classifier
    exec(open('vgg_room_classifier_train.py').read())

    # Classify rooms
    exec(open('vgg_room_classifier_test.py').read())
    exec(open('data_process.py').read())

    # Extract image features using googleNet
    exec(open('googlenet_feature_extract.py').read())

    #  Classify luxury level
    exec(open('lux_level_classification.py').read())
    exec(open('vgg_lux_level_classification.py').read())

    # run regression model
    exec(open('Regression.py').read())