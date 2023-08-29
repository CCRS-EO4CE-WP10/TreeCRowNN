import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import models, layers
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    concatenate,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


def crop_image(
    img, tile_size
):  # this function defines the central extent and crops input to match
    y, x = img.shape
    cropx = round(int(x / tile_size), 0) * tile_size
    cropy = round(int(y / tile_size), 0) * tile_size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    print(startx, startx + cropx, starty, starty + cropy)
    return img[starty : starty + cropy, startx : startx + cropx]


def mod1(lr, spe, u, u2, u3):
    in1 = Input(shape=(128, 128, 3))
    conv1 = Conv2D(u, (3, 3), activation="relu", padding="same")(in1)
    BN1 = BatchNormalization()(conv1)
    conv2 = Conv2D(u, (3, 3), activation="relu", padding="same")(BN1)
    BN2 = BatchNormalization()(conv2)
    small1 = Conv2D(u, (3, 3), activation="relu", padding="same")(BN2)
    BN3 = BatchNormalization()(small1)
    small2 = Conv2D(u, (3, 3), activation="relu", padding="same")(BN3)
    BN4 = BatchNormalization()(small2)
    small3 = MaxPooling2D((2, 2))(BN4)
    small4 = Dropout(0.2)(small3)
    small5 = Conv2D(u, (3, 3), activation="relu", padding="same")(small4)
    BN5 = BatchNormalization()(small5)
    small6 = Conv2D(u, (3, 3), activation="relu", padding="same")(BN5)
    BN6 = BatchNormalization()(small6)
    small7 = Dropout(0.2)(BN6)
    small8 = Conv2D(u, (3, 3), activation="relu", padding="same")(small7)
    BN7 = BatchNormalization()(small8)
    small9 = Conv2D(u, (3, 3), activation="relu", padding="same")(BN7)
    BN8 = BatchNormalization()(small9)
    small10 = MaxPooling2D((2, 2))(BN8)
    small11 = Dropout(0.2)(small10)
    med1 = Conv2D(u, (7, 7), activation="relu", padding="same")(BN2)
    BN9 = BatchNormalization()(med1)
    med2 = Conv2D(u, (7, 7), activation="relu", padding="same")(BN9)
    BN10 = BatchNormalization()(med2)
    med3 = MaxPooling2D((2, 2))(BN10)
    med4 = Dropout(0.2)(med3)
    med5 = Conv2D(u, (7, 7), activation="relu", padding="same")(med4)
    BN11 = BatchNormalization()(med5)
    med6 = Conv2D(u, (7, 7), activation="relu", padding="same")(BN11)
    BN12 = BatchNormalization()(med6)
    med7 = Dropout(0.2)(BN12)
    med8 = Conv2D(u, (7, 7), activation="relu", padding="same")(med7)
    BN13 = BatchNormalization()(med8)
    med9 = Conv2D(u, (7, 7), activation="relu", padding="same")(BN13)
    BN14 = BatchNormalization()(med9)
    med10 = MaxPooling2D((2, 2))(BN14)
    med11 = Dropout(0.2)(med10)
    big1 = Conv2D(u, (15, 15), activation="relu", padding="same")(BN2)
    BN15 = BatchNormalization()(big1)
    big2 = Conv2D(u, (15, 15), activation="relu", padding="same")(BN15)
    BN16 = BatchNormalization()(big2)
    big3 = MaxPooling2D((2, 2))(BN16)
    big4 = Dropout(0.2)(big3)
    big5 = Conv2D(u, (15, 15), activation="relu", padding="same")(big4)
    BN17 = BatchNormalization()(big5)
    big6 = Conv2D(u, (15, 15), activation="relu", padding="same")(BN17)
    BN18 = BatchNormalization()(big6)
    big7 = Dropout(0.2)(BN18)
    big8 = Conv2D(u, (15, 15), activation="relu", padding="same")(big7)
    BN19 = BatchNormalization()(big8)
    big9 = Conv2D(u, (15, 15), activation="relu", padding="same")(BN19)
    BN20 = BatchNormalization()(big9)
    big10 = MaxPooling2D((2, 2))(BN20)
    big11 = Dropout(0.2)(big10)
    concat1 = tf.keras.layers.Concatenate()([small11, med11, big11])
    FC1 = Conv2D(u2, (1, 1), activation="relu", padding="valid")(concat1)
    BN21 = BatchNormalization()(FC1)
    FC2 = Conv2D(u, (5, 5), activation="relu", padding="valid")(BN21)
    BN22 = BatchNormalization()(FC2)
    FC3 = Conv2D(u, (5, 5), activation="relu", padding="valid")(BN22)
    BN23 = BatchNormalization()(FC3)
    FC4 = Conv2D(u, (5, 5), activation="relu", padding="valid")(BN23)
    BN24 = BatchNormalization()(FC4)
    flat = Flatten()(BN24)
    dense1 = keras.layers.Dense(u3, activation="relu")(flat)
    BN25 = BatchNormalization()(dense1)
    dense2 = keras.layers.Dense(u3, activation="relu")(BN25)
    BN26 = BatchNormalization()(dense2)
    dense3 = keras.layers.Dense(u3, activation="relu")(BN26)
    BN27 = BatchNormalization()(dense3)
    dense4 = keras.layers.Dense(u3, activation="relu")(BN27)
    BN28 = BatchNormalization()(dense4)
    dense5 = keras.layers.Dense(u3, activation="relu")(BN28)
    BN29 = BatchNormalization()(dense5)
    out1 = keras.layers.Dense(1, activation="linear")(BN29)
    model = Model(inputs=[in1], outputs=[out1])
    model.compile(
        loss="MeanAbsoluteError",
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError(),
        ],
    )
    return model


def get_heatmap(
    in1, model, last_conv, tile_size, padding, normalize_heatmap, pred_index=None
):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape(persistent=False) as tape:
        last_conv_output, preds = grad_model(in1)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_output = last_conv_output[0]
        heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = cv2.resize(
            heatmap.numpy(), (tile_size + (2 * padding), tile_size + (2 * padding))
        )
        if padding != 0:
            heatmap = heatmap[padding:-padding, padding:-padding]
        if normalize_heatmap == "local":
            normalize_activations(heatmap)
    return heatmap


def get_predict(patch):
    import numpy as np

    prediction = np.mean(patch)
    prediction = np.rint(prediction)
    if prediction < 0.0:
        prediction = 0.0
    return prediction


def create_image(
    img,
    blank_img,
    blank_img2,
    tile_size,
    model,
    chan,
    padding,
    minval,
    maxval,
    batch_size,
    target_layer,
    generate_heatmap,
    normalize_heatmap,
):
    y, x = img.shape[:-1]
    nblocks_row = int(y // tile_size)
    nblocks_col = int(x // tile_size)
    total_blocks = nblocks_row * nblocks_col
    print("TOTAL BLOCKS : ", total_blocks)
    print("BATCH SIZE : ", batch_size)
    row_list = range(0, nblocks_row)
    col_list = range(0, nblocks_col)
    n = 0
    dataset = []
    coords = []
    for i in row_list:
        for j in col_list:
            row_start = i
            row_start = row_start * tile_size
            row_end = row_start + tile_size
            col_start = j
            col_start = col_start * tile_size
            col_end = col_start + tile_size
            block = np.array([row_start, row_end, col_start, col_end]).astype("int32")
            patch = img[row_start:row_end, col_start:col_end]
            patch = tf.cast(patch, tf.float32) / maxval
            patch = tf.reshape(patch, shape=[1, tile_size, tile_size, chan])
            patch = tf.keras.layers.ZeroPadding2D(padding=padding)(patch)
            if generate_heatmap == True:
                heatmap = get_heatmap(
                    patch, model, target_layer, tile_size, padding, normalize_heatmap
                )
                blank_img2[block[0] : block[1], block[2] : block[3]] = heatmap
            coords.append([i, j])
            dataset.append(patch)
            n += 1
            if len(dataset) >= batch_size:
                dataset2 = tf.data.Dataset.from_tensor_slices(dataset)
                result = model.predict(dataset2)
                for k in range(len(result)):
                    predict = get_predict(result[k])
                    blank_img[coords[k][0], coords[k][1]] = predict
                coords = []
                dataset = []
            if str(n)[-4:] == "0000":
                t2 = time.time() - t1
                print(
                    "{} pixels processed in {}, {}% done".format(
                        n, t2, str("%.2f" % (n / total_blocks * 100))
                    )
                )
    print("FINAL PIXELS")
    if len(dataset) > 0:
        dataset2 = tf.data.Dataset.from_tensor_slices(dataset)
        result = model.predict(dataset2)
        for k in range(len(result)):
            predict = get_predict(result[k])
            blank_img[coords[k][0], coords[k][1]] = predict
    t3 = time.time()
    total_time = t3 - t1
    return total_time
