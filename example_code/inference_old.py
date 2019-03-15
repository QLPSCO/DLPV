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
"""
config = tf.ConfigProto(
    device_count = {'GPU': 1}
)
sess = tf.Session(config=config)
"""


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#inception_base = applications.ResNet50(weights='imagenet', include_top=False)
inception_base = applications.VGG16(weights="imagenet", include_top=False)

for layer in inception_base.layers:
    layer.trainable = False

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(1, activation='sigmoid')(x)
# create the full network so we can train on it
custom_resnet_model = Model(inputs=inception_base.input, outputs=predictions)

print(custom_resnet_model.summary())

#custom_resnet_model.load_weights('resnet_gpu_tf.h5')
custom_resnet_model.load_weights('vgg_gpu_tf.h5')
"""
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
"""
datagen = ImageDataGenerator()

generator = datagen.flow_from_directory(
        'data/inference_100',
        target_size=(150, 150),
        batch_size=1,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

t0 = time.time()
#probabilities = custom_resnet_model.predict_generator(generator, steps=500, workers=4, use_multiprocessing=True, verbose=1)
probabilities = custom_resnet_model.predict_generator(generator, steps=500,  use_multiprocessing=True, verbose=1)
t1 = time.time()
print(t1-t0)

"""

# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('model1.h5')

# predicting images
img = image.load_img('a.png', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print(classes)
"""
