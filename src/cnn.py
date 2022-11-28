import numpy as np
import cv2 as cv
import imutils
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from src import files

def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    # handle normalizing the histogram if we are using OpenCV 2.4.X
    # if imutils.cv():
    #     hist = cv.normalize(hist)
    # otherwise, perform "in place" normalization in OpenCV 3 (I
    # personally hate the way this is done
    # else:
    cv.normalize(hist, hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv.resize(image, size).flatten()

def create_model(data_set, data_set_label, end_layer, path, rounds):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                            input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(550, activation="relu"),  # Camada oculta
        tf.keras.layers.Dropout(0.1, seed=2019),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dropout(0.3, seed=2019),
        tf.keras.layers.Dense(300, activation="relu"),
        tf.keras.layers.Dropout(0.4, seed=2019),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dropout(0.2, seed=2019),
        tf.keras.layers.Dense(end_layer, activation="softmax" if end_layer == 5 else "sigmoid")  # Camada de saida
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f"{path}/../weight.hdf5", monitor='loss', verbose=end_layer,
    save_best_only=True, mode='auto', period=1, save_weights_only=True)

    history = model.fit(
        x=data_set,
        y=data_set_label,
        validation_split=0.1,
        epochs=rounds,
        callbacks=[checkpoint])

    model.save(path)

def train_model(model, data_set, data_set_label, path, end_layer=5, rounds=5):

    checkpoint = ModelCheckpoint(f"{path}/../{end_layer}_weight.hdf5", monitor='loss', verbose=end_layer,
    save_best_only=True, mode='auto', period=1, save_weights_only=True)

    history = model.fit(
        x=data_set,
        y=data_set_label,
        validation_split=0.1,
        epochs=rounds,
        callbacks=[checkpoint])

    model.save(path)

def load_models():
    try:
        reloaded_binary = tf.keras.models.load_model('./ClsKLData/kneeKL224/model/2')
        reloaded_binary.summary()
        reloaded = tf.keras.models.load_model('./ClsKLData/kneeKL224/model/5')
        reloaded.summary()
        return reloaded, reloaded_binary
    except:
        binary = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                                input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(550, activation="relu"),  # Camada oculta
            tf.keras.layers.Dropout(0.1, seed=2019),
            tf.keras.layers.Dense(400, activation="relu"),
            tf.keras.layers.Dropout(0.3, seed=2019),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dropout(0.4, seed=2019),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dropout(0.2, seed=2019),
            tf.keras.layers.Dense(2, activation="softmax" if 2 == 5 else "sigmoid")  # Camada de saida
        ])

        binary.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        five_classes = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
                                input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(550, activation="relu"),  # Camada oculta
            tf.keras.layers.Dropout(0.1, seed=2019),
            tf.keras.layers.Dense(400, activation="relu"),
            tf.keras.layers.Dropout(0.3, seed=2019),
            tf.keras.layers.Dense(300, activation="relu"),
            tf.keras.layers.Dropout(0.4, seed=2019),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dropout(0.2, seed=2019),
            tf.keras.layers.Dense(5, activation="softmax" if 5 == 5 else "sigmoid")  # Camada de saida
        ])

        five_classes.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'])

        return five_classes, binary

evaluation = []
history = {}

def evaluete_model(model, verbose_=5):
    binary_ = False if verbose_ == 5 else True
    verbose_class = 2 if verbose_ == 2 else 5
    data_set, data_set_label = files.read_data_set_val('./ClsKLData/kneeKL224/data_set_val/', amount=5, binary=binary_)
    tr_loss, tr_acc = model.evaluate(data_set, data_set_label, verbose=verbose_)
    print(tr_loss, tr_acc)
    predictions = model.predict(data_set)

    predicted = []
    for prediction in predictions:
        predicted.append(np.argmax(prediction))

    predicted = np.array(predicted)

    print(predicted)
    print(data_set_label)

    confused = tf.math.confusion_matrix(
        labels=data_set_label,
        predictions=predicted,
        num_classes=verbose_class
    )

    evaluation.clear()
    evaluation.append(f'loss={tr_loss*100} | acc={tr_acc*100}')
    evaluation.append('predicted_data')
    evaluation.append(predicted)
    evaluation.append('label')
    evaluation.append(data_set_label)
    evaluation.append('confusion matrix')
    evaluation.append(confused)

    print(confused)

