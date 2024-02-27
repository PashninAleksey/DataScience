from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
import pandas as pd

import numpy as np

def load_train(path):
    
    labels = pd.read_csv(path + 'labels.csv')

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=90,
        validation_split=0.25,
        rescale=1/255.
    )
    result = datagen.flow_from_dataframe(
        dataframe = labels,
        directory = path+'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='training',
        seed=123
    )
    return result

def load_test(path):
    
    labels = pd.read_csv(path + 'labels.csv')

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=90,
        validation_split=0.25,
        rescale=1/255.
    )
    result = datagen.flow_from_dataframe(
        dataframe = labels,
        directory = path+'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=32,
        class_mode='raw',
        subset='validation',
        seed=123
    )
    return result


def create_model(input_shape):
    optimizer = Adam(learning_rate=0.00001)
    backbone = ResNet50(input_shape=input_shape,
                    weights='/datasets/keras_models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                    include_top=False)
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=15,
               steps_per_epoch=None, validation_steps=None):


    model.fit(train_data,
          validation_data=test_data,
          steps_per_epoch=len(train_data),
          validation_steps=len(test_data),
          epochs=epochs,
          verbose=2)

    return model