#!/usr/bin/env python
# coding: utf-8

# # SSD BNN Jet Detection Training

# In[46]:


# Import GPU libs

import setGPU
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# Other imports

import numpy as np
import simplejson as json

from math import ceil


# In[3]:


# Set presentation settings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as tick

from matplotlib import cm
from matplotlib.colors import SymLogNorm
from mpl_toolkits import mplot3d

matplotlib.rcParams["figure.figsize"] = (16.0, 6.0)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Anonymous Pro for Powerline']

matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.top"] = True
matplotlib.rcParams["axes.spines.right"] = True
matplotlib.rcParams["axes.spines.bottom"] = True
matplotlib.rcParams["axes.labelsize"] = 16
matplotlib.rcParams["axes.titlesize"] = 14

matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["xtick.major.size"] = 10
matplotlib.rcParams["ytick.major.size"] = 10
matplotlib.rcParams["xtick.minor.size"] = 5
matplotlib.rcParams["ytick.minor.size"] = 5
matplotlib.rcParams["xtick.minor.visible"] = True

matplotlib.rcParams["lines.linewidth"] = 2

matplotlib.rcParams["legend.fontsize"] = 14

with open('./data/palette.json') as json_file:
    color_palette = json.load(json_file)


# In[4]:


import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[5]:


from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import Adam
from keras.models import load_model


# In[6]:


from ssd.generator import DataGenerator
from ssd.keras_ssd7_bnn import build_model
from ssd.keras_ssd_loss import SSDLoss
from ssd.ssd_input_encoder import SSDInputEncoder
from ssd.ssd_output_decoder import decode_detections


# In[7]:


# Model configuration parameters

SAVE_PATH = '/data/adpol'
MODEL_NAME = 'ceva-cms-jet-ssd-bnn'
DATA_SOURCE = '/eos/user/a/adpol/ceva'
TRAINING_EPOCHS = 1000
SPLIT = [0.1, 0.1, 0.1]
MAX_EVENTS = None

classes = ['background', 'b', 'h', 'W', 't', 'q']

img_height = 452 # Pixel height
img_width = 340 # Pixel width
img_channels = 1 # Number of channels
n_classes = 4 # Number of target classes

# Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_mean = None 
intensity_range = None

# An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
scales = [0.16, 0.4, 0.6, 0.8, 0.96]

# The list of aspect ratios for the anchor boxes
aspect_ratios = [1.0]
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = True # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size


# In[8]:


# Compile the model

K.clear_session()

model = build_model(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    mode='training',
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

# Instantiate an Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)


# In[9]:


# Training data generator

hdf5_dataset_paths_list = []

for i in np.arange(0, 10*SPLIT[0], dtype=np.int16):
    for j in ['bb', 'tt', 'WW', 'hh']:
        hdf5_dataset_paths_list.append('%s/RSGraviton_%s_NARROW_%s.h5' % (DATA_SOURCE, j, i))

train_dataset = DataGenerator(hdf5_dataset_paths=hdf5_dataset_paths_list, max_size=MAX_EVENTS)


# In[10]:


# Training data generator

hdf5_dataset_paths_list = []

for i in np.arange(10*SPLIT[0], 10*(SPLIT[0]+SPLIT[1]), dtype=np.int16):
    for j in ['bb', 'tt', 'WW', 'hh']:
        hdf5_dataset_paths_list.append('%s/RSGraviton_%s_NARROW_%s.h5' % (DATA_SOURCE, j, i))

val_dataset = DataGenerator(hdf5_dataset_paths=hdf5_dataset_paths_list, max_size=MAX_EVENTS)


# In[11]:


train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))


# In[12]:


# Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

batch_size = 100

predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                   model.get_layer('classes5').output_shape[1:3],
                   model.get_layer('classes6').output_shape[1:3],
                   model.get_layer('classes7').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)


# In[13]:


# Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'})

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'})


# ## Model training

# In[14]:


# Check if GPU is available

print('GPU Available? %s' % (len(K.tensorflow_backend._get_available_gpus()) > 0))


# In[15]:


# Define callbacks

model_checkpoint = ModelCheckpoint(filepath='%s/%s.h5' % (SAVE_PATH, MODEL_NAME),
                                   monitor='val_loss',
                                   verbose=0,
                                   save_best_only=True,
                                   save_weights_only=True,
                                   mode='auto',
                                   period=1)

csv_logger = CSVLogger(filename='%s/%s.csv' % (SAVE_PATH, MODEL_NAME),
                       separator=',',
                       append=False)

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0.0,
                               patience=10,
                               verbose=0)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.2,
                                         patience=8,
                                         verbose=0,
                                         min_delta=0.001,
                                         cooldown=0,
                                         min_lr=0.00001)

callbacks = [model_checkpoint,
             csv_logger,
             early_stopping,
             reduce_learning_rate]


# In[16]:


history = model.fit_generator(generator=train_generator,
                              use_multiprocessing=False,
                              validation_data=train_generator,
                              steps_per_epoch=int(np.floor(train_dataset_size/batch_size)),
                              validation_steps=int(np.floor(train_dataset_size/batch_size)),
                              epochs=TRAINING_EPOCHS,
                              workers=0,
                              callbacks=callbacks)


# In[48]:


model.layers[6]


# In[ ]:




