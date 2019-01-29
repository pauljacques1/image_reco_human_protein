import pandas as pd
import numpy as np

from keras.models import load_model
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


df_submission = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/sample_submission.csv')
print(df_submission.head())
print(df_submission.info())

for i in range(0,28):
    model = load_model('/kaggle/input/trained-models/trained_resnet 2/trained_RESNET/trained_resnet50_kaggle_{}.h5'.format(i))
    print('model {} loaded'.format(i))
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        directory ="/kaggle/input/trytwenty/trytwenty", 
        target_size=(512, 512),
        color_mode="rgb",
        batch_size=1,
        class_mode='categorical'
        )
    
    #test_generator.reset()

    pred = model.predict_generator(test_generator,verbose=1)
    predicted_class_indices=np.argmax(pred,axis=1)
    print(predicted_class_indices)
    filenames=[(filename.split("/")[1]).split(".png")[0] for filename in test_generator.filenames]
    
    results=pd.DataFrame({'Id':filenames ,
                      str(i):predicted_class_indices})
    print(results.head())
    df_submission = df_submission.merge(results, on = 'Id')

    #results.to_csv("results.csv",index=False)
    print(df_submission.head())
    

df_submission.to_csv("df_submission.csv",index=False)
