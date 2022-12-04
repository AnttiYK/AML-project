import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import random
import matplotlib.pyplot as plt


'split data to test and train sets'
def split_data(data):
    split_size = 0.8
    split_index = int(len(data) * split_size)
    train_ds = data[:split_index]
    test_ds = data[split_index+1:]
    return train_ds, test_ds

'predict results from image'
def predict(model, image):
    return model.predict(np.expand_dims(image, axis=0))

'learning analyzis'
def analyzis(history, n_epochs):
    loss = history.history['loss']
    sm_loss   = history.history['sm_head_loss']
    fm_loss = history.history['fm_head_loss']
    sm_acc = history.history['sm_head_accuracy']
    fm_acc = history.history['fm_head_accuracy']
    val_loss    = history.history['val_loss']
    val_sm_loss = history.history['val_sm_head_loss']
    val_fm_loss = history.history['val_fm_head_loss']
    val_sm_acc = history.history['val_sm_head_accuracy']
    val_fm_acc = history.history['val_fm_head_accuracy']
    xc         = range(n_epochs)
    plt.figure()
    plt.suptitle("Training analysis")
    l = plt.subplot(4,3,1)
    plt.plot(xc, loss)
    l.set_title('training loss')

    sl = plt.subplot(4,3,2)
    plt.plot(xc,sm_loss)
    sl.set_title("sm training loss")

    fl = plt.subplot(4,3,3)
    plt.plot(xc, fm_loss)
    fl.set_title('fm training loss')

    sa = plt.subplot(4,3,4)
    plt.plot(xc, sm_acc)
    sa.set_title('sm training accuracy')

    fa = plt.subplot(4,3,5)
    plt.plot(xc, fm_acc)
    fa.set_title('fm training accuracy')

    vl = plt.subplot(4,3,6)
    plt.plot(xc, val_loss)
    vl.set_title('validation loss')

    vsl = plt.subplot(4,3,7)
    plt.plot(xc, val_sm_loss)
    vsl.set_title('sm validation loss')

    vfl = plt.subplot(4,3,8)
    plt.plot(xc, val_fm_loss)
    vfl.set_title('fm validation loss')

    vsa = plt.subplot(4,3,9)
    plt.plot(xc, val_sm_acc)
    vsa.set_title('sm validation accuracy')    
    
    vfa = plt.subplot(4,3,10)
    plt.plot(xc, val_fm_acc)
    vfa.set_title('fm validation accuracy')
    plt.tight_layout()
    plt.show()


'model architecture and training'
def multi_task_model(images_train, images_test, face_train, face_test, smile_train, smile_test, scaled_size, n_epochs):
    'input layer'
    input_layer = keras.layers.Input(shape=(scaled_size, scaled_size, 3))
    
    'base branch'
    base_model = keras.layers.experimental.preprocessing.Rescaling(1./255, name='bm1')(input_layer)
    base_model = keras.layers.Conv2D(16, 3, padding='same', activation='relu', name='bm2')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm3')(base_model)
    base_model = keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='bm4')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm5')(base_model)
    base_model = keras.layers.Conv2D(64, 3, padding='same', activation='relu', name='bm6')(base_model)
    base_model = keras.layers.MaxPooling2D(name = 'bm7')(base_model)
    base_model = keras.layers.Flatten(name='bm8')(base_model)

    'smile branch'
    smile_model = keras.layers.Dense(128, activation='relu', name='sm1')(base_model)
    smile_model = keras.layers.Dense(2, name='sm_head')(smile_model)

    'face branch'
    face_model = keras.layers.Dense(128, activation = 'relu', name = 'fm1')(base_model)
    face_model = keras.layers.Dense(64, activation = 'relu', name = 'fm2')(face_model)
    face_model = keras.layers.Dense(32, activation = 'relu', name = 'fm3')(face_model)
    face_model = keras.layers.Dense(4, activation='sigmoid', name = 'fm_head')(face_model)

    'model structure'
    model = keras.Model(input_layer, outputs=[smile_model, face_model])

    'losses functions for both branches'
    losses = {"sm_head": keras.losses.SparseCategoricalCrossentropy(from_logits=True), "fm_head": keras.losses.MSE}

    'compile model'
    model.compile(loss = losses, optimizer = 'Adam', metrics=['accuracy'])

    'separate validation targets for both branches'
    trainTargets = {
        "sm_head": smile_train,
        "fm_head": face_train
    }
    testTargets = {
        "sm_head": smile_test,
        "fm_head": face_test
    }

    'fit model'
    history = model.fit(images_train, trainTargets, validation_data = (images_test, testTargets), epochs = n_epochs, shuffle = True, verbose = 1)

    analyzis(history, n_epochs)
    
    'save model'
    model.save("cnn_model")

