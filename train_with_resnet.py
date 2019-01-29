import pandas as pd
import os

from keras.layers import Dense
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical
import tensorflow as tf

PATH = 'train_merged'

#df = pd.read_csv(os.path.join(PATH,'trainLabels.csv'))
df = pd.read_csv('multiLabels.csv')

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512,512,3))

x = base_model.output
x = Flatten(name='flatten')(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[0:141]:
    layer.trainable = False

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_generator=datagen.flow_from_dataframe(dataframe=df, 
                                            directory='train_merged',
                                            x_col="Id", 
                                            y_col="col_0", 
                                            has_ext=False, 
                                            class_mode="categorical", 
                                            target_size=(512,512), 
                                            batch_size=100)


# fits the model on batches with real-time data augmentation:
model.fit_generator(train_generator,
                    steps_per_epoch=df.shape[0] / 100, epochs=epochs)


model.save('trained_resnet50_kaggle_0.h5')