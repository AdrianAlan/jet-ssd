import cv2
import h5py
import numpy as np
import torch

class CalorimeterJetDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 hdf5_dataset=None,
                 labels_output_format=('class_id',
                                       'xmin',
                                       'ymin',
                                       'xmax',
                                       'ymax',
                                       'pt')):
        
        self.hdf5_dataset = hdf5_dataset
        self.labels_output_format = labels_output_format
        
        self.height = 452 # Height of the input images
        self.width = 340 # Width of the input images
        self.channels = 2 # Number of color channels of the input images
        
        # This dictionary is for internal use
        self.labels_format={'class_id': labels_output_format.index('class_id'),
                            'xmin': labels_output_format.index('xmin'),
                            'ymin': labels_output_format.index('ymin'),
                            'xmax': labels_output_format.index('xmax'),
                            'ymax': labels_output_format.index('ymax'),
                            'pt': labels_output_format.index('pt')}

        self.labels = self.hdf5_dataset['labels']
        self.calorimeter = self.hdf5_dataset['calorimeter']
        
        self.dataset_size = len(self.calorimeter)
        
    def __getitem__(self, index):
        redo = True
        while redo:
            label_raw = np.asarray([self.labels[index]], dtype=np.float32)
            label_reshaped = label_raw.reshape(-1, 6)
            mask = label_reshaped[:, 5] > 100
            label_reshaped = label_reshaped[:, :-1] # Remove PT
            label_reshaped = label_reshaped[:, [1,2,3,4,0]] # Class Last
            label_reshaped[:, 0] = label_reshaped[:, 0] / float(self.width) # x_min
            label_reshaped[:, 2] = label_reshaped[:, 2] / float(self.width) # x_max
            label_reshaped[:, 1] = label_reshaped[:, 1] / float(self.height) # y_min
            label_reshaped[:, 3] = label_reshaped[:, 3] / float(self.height) # y_max
            label_reshaped[:, 4] = 0.#label_reshaped[:, 4] - 1. # label
            label_reshaped = label_reshaped[mask]
            label_reshaped = np.asarray(label_reshaped, dtype=np.float32)
            label_reshaped = torch.from_numpy(label_reshaped)

            calorimeter = self.calorimeter[index]
            calorimeter = calorimeter.reshape(self.channels, self.height, self.width)
            calorimeter_max = calorimeter.reshape(
                self.channels, -1).max(axis=1).reshape(self.channels, 1, 1) + 1e-12
            calorimeter = calorimeter / calorimeter_max
            # calorimeter = np.rollaxis(calorimeter, 0, 3) # channels last
            calo = np.zeros((2,300,300))
            calo[0] = cv2.resize(calorimeter[0], dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
            calo[1] = cv2.resize(calorimeter[1], dsize=(300, 300), interpolation=cv2.INTER_CUBIC)

            calo = np.asarray(calo, dtype=np.float32)
            calo = torch.from_numpy(calo)
            
            if label_reshaped.shape[0]:
                redo = False

            index += 1

        return calo, label_reshaped

    def __len__(self):
        return self.dataset_size
