from __future__ import division

import h5py
import time
import numpy as np
import sklearn.utils
import sys

from tqdm import trange

class DatasetError(Exception):
    pass

class DataGenerator:

    def __init__(self,
                 hdf5_dataset_path=None,
                 max_size=None,
                 labels_output_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax'),
                 verbose=True):
        
        self.hdf5_dataset_path = hdf5_dataset_path
        self.labels_output_format = labels_output_format
        
        # This dictionary is for internal use
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax')} 
        
        # We start with dataset size zero
        self.dataset_size = 0 
        
        self.height = 452 # Height of the input images
        self.width = 340 # Width of the input images
        self.channels = 2 # Number of color channels of the input images
        
#         self.height = 300 # Height of the input images
#         self.width = 480 # Width of the input images
#         self.channels = 3 # Number of color channels of the input images
        
        self.dataset_labels, self.dataset_labels_encoded, self.image_ids = [], [], None
        
        self.load_hdf5_dataset(max_size)
            
    def load_hdf5_dataset(self, max_size):
       
        # Load given h5 file
        self.hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'r')
        
        # Store size of the dataset
        if max_size:
            size = max_size
        else:
            size = len(self.hdf5_dataset['calorimeter'])

        self.dataset_size = size
        self.image_ids = np.arange(size, dtype=np.int32)
        
        # Process labels
        print("Processing labels for {}".format(self.hdf5_dataset_path))
        labels = self.hdf5_dataset['labels']
        shapes = self.hdf5_dataset['label_shapes']
            
        tr = trange(len(labels), file=sys.stdout)
        tr.set_description('Loading labels')

        for i in tr:
            label_processed = labels[i].reshape(shapes[i])#[:,:-1]
            self.dataset_labels.append(label_processed)
            
        self.dataset_labels = np.array(self.dataset_labels)
                   
    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'}):
        
        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        # Shuffle the indices
        if shuffle:
            shuffled_objects = sklearn.utils.shuffle(self.image_ids)
            
        # Generate mini batches.
        current = 0
      
        while True:
           
            calorimeter, labels, ids = [], [], []
            
            if current >= self.dataset_size:
                current = 0
                
            if shuffle and current == 0:
                shuffled_objects = sklearn.utils.shuffle(self.image_ids)
                
            if shuffle:
                batch_indices = shuffled_objects[current:(current + batch_size)]
            else:
                batch_indices = self.image_ids[current:(current + batch_size)]
                
            batch_indices = sorted(batch_indices)
           
            calorimeter = self.hdf5_dataset['calorimeter'][batch_indices]
            # List to np array
            calorimeter = np.vstack(calorimeter)
            
            # Set to nominal shape
            calorimeter = calorimeter.reshape(batch_size, self.channels, self.height, self.width)
            
            # Normaliztion: Max in ECAL in HCAL separately
            calorimeter_max = calorimeter.reshape(batch_size*self.channels, -1).max(axis=1)
            calorimeter_max = calorimeter_max.reshape(batch_size, self.channels, 1, 1)
            calorimeter = calorimeter / calorimeter_max
            
            # Extend to 3d
            calorimeter = np.repeat(calorimeter, 2, axis=0).reshape(batch_size, -1, self.height, self.width)[:,:-1]
            
            # Move channels last
            calorimeter = np.rollaxis(calorimeter, 1, 4)
            
            labels = self.dataset_labels[batch_indices]
            
            if 'image_ids' in returns:
                ids = np.array(batch_indices)

            current += batch_size

            if (calorimeter.size == 0):
                raise DatasetError("Empty batch produced. Cause: varying size or channels?")

            if label_encoder is not None:
                batch_y_encoded = label_encoder(labels, diagnostics=False)
            else:
                batch_y_encoded = None

            # Compose the output.
            ret = []
            if 'image_ids' in returns: ret.append(ids)
            if 'processed_images' in returns: ret.append(calorimeter)
            if 'processed_labels' in returns: ret.append(labels)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)

            start = time.time()
            yield ret

    def get_dataset_size(self):
        return self.dataset_size
