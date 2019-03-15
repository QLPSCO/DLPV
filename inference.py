from keras.preprocessing import image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras import applications
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
import tensorflow as tf
import time
import os

import theano
os.environ["OMP_NUM_THEADS"] = "4"
theano.config.openmp = True

"""
config = tf.ConfigProto(
    device_count = {'GPU': 1}
)
sess = tf.Session(config=config)
"""

# recreate network from training and fill in with saved weights for inference
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

base = applications.ResNet50(weights='imagenet', include_top=False)
#base = applications.VGG16(weights="imagenet", include_top=False)

for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
custom_model = Model(inputs=base.input, outputs=predictions)

# If using resnet uncomment first line, if using vgg uncomment second line
custom_model.load_weights('resnet_gpu_tf.h5')
#custom_model.load_weights('vgg_gpu_tf.h5')
datagen = ImageDataGenerator()

generator = datagen.flow_from_directory(
        'data/inference_100',
        target_size=(150, 150),
        batch_size=1,
        class_mode=None,
        shuffle=False)

t0 = time.time()
#probabilities = custom_model.predict_generator(generator, steps=500, workers=4, use_multiprocessing=True, verbose=1)
probabilities = custom_model.predict_generator(generator, steps=500,  use_multiprocessing=True, verbose=1)
t1 = time.time()
print(t1-t0)

