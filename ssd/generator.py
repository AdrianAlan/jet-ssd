import cv2
import h5py
import numpy as np
import torch


class CalorimeterJetDataset(torch.utils.data.Dataset):

    def __init__(self, hdf5_dataset=None,
                 l_format=('class_id', 'xmin', 'ymin', 'xmax', 'ymax', 'pt')):

        self.hdf5_dataset = hdf5_dataset
        self.l_format = l_format

        self.offset_phi = 46
        self.height = 360 + 2 * self.offset_phi  # Height of the input images
        self.width = 340  # Width of the input images
        self.channels = 2  # Number of color channels of the input images
        # This dictionary is for internal use
        self.labels_format = {'class_id': l_format.index('class_id'),
                              'xmin': l_format.index('xmin'),
                              'ymin': l_format.index('ymin'),
                              'xmax': l_format.index('xmax'),
                              'ymax': l_format.index('ymax'),
                              'pt': l_format.index('pt')}

        self.labels = self.hdf5_dataset['labels']

        self.ecal_energy = self.hdf5_dataset['ecal_energy']
        self.ecal_phi = self.hdf5_dataset['ecal_phi']
        self.ecal_eta = self.hdf5_dataset['ecal_eta']

        self.hcal_energy = self.hdf5_dataset['hcal_energy']
        self.hcal_phi = self.hdf5_dataset['hcal_phi']
        self.hcal_eta = self.hdf5_dataset['hcal_eta']

        self.dataset_size = len(self.labels)

    def process_labels(self, labels_raw):
        label_reshaped = labels_raw.reshape(-1, 6)  # Reshape labels
        label_reshaped = label_reshaped[:, :-1]  # Remove PT
        label_reshaped = label_reshaped[:, [1, 2, 3, 4, 0]]  # Class last

        # Set fractional coordinates
        label_reshaped[:, 0] = label_reshaped[:, 0] / float(self.width)
        label_reshaped[:, 2] = label_reshaped[:, 2] / float(self.width)
        label_reshaped[:, 1] = label_reshaped[:, 1] / float(self.height)
        label_reshaped[:, 3] = label_reshaped[:, 3] / float(self.height)
        label_reshaped[:, 4] = 0.  # label_reshaped[:, 4] - 1. # label

        return torch.from_numpy(label_reshaped)

    def process_images(self, indices_phi, indices_eta, energy, cal='ECAL'):
        energy = energy / np.max(energy)

        if cal == 'ECAL':
            pixels = np.zeros((360, 340))
        if cal == 'HCAL':
            pixels = np.zeros((360, 204))

        pixels[indices_phi, indices_eta] = energy

        if cal == 'HCAL':
            pixels = np.hstack((pixels[:, :85],
                                pixels[:, 85:-85].repeat(5, axis=1),
                                pixels[:, -85:]))

        # Extend image top and bottom by a radius of the biggest jet
        # Prevents cropped boxes in y plane
        pixels = np.vstack((pixels[-self.offset_phi:, :],
                            pixels,
                            pixels[:self.offset_phi, :]))

        return pixels

    def __getitem__(self, index):

        # Repeat for events with no jets: Causes problems during training
        repeat = True

        while repeat:

            # Set labels
            labels_raw = np.asarray([self.labels[index]], dtype=np.float32)
            labels = self.process_labels(labels_raw)

            # Load calorimeter
            indices_ecal_phi = np.asarray(self.ecal_phi[index], dtype=np.int16)
            indices_ecal_eta = np.asarray(self.ecal_eta[index], dtype=np.int16)
            ecal_energy = np.asarray(self.ecal_energy[index], dtype=np.float32)

            ecal_pixels = self.process_images(indices_ecal_phi,
                                              indices_ecal_eta,
                                              ecal_energy,
                                              cal='ECAL')

            indices_hcal_phi = np.asarray(self.hcal_phi[index], dtype=np.int16)
            indices_hcal_eta = np.asarray(self.hcal_eta[index], dtype=np.int16)
            hcal_energy = np.asarray(self.hcal_energy[index], dtype=np.float32)

            hcal_pixels = self.process_images(indices_hcal_phi,
                                              indices_hcal_eta,
                                              hcal_energy,
                                              cal='HCAL')

            # calorimeter = np.rollaxis(calorimeter, 0, 3) # channels last
            calo = np.zeros((2, 340, 340))
            calo[0] = cv2.resize(ecal_pixels, dsize=(340, 340),
                                 interpolation=cv2.INTER_CUBIC)
            calo[1] = cv2.resize(hcal_pixels, dsize=(340, 340),
                                 interpolation=cv2.INTER_CUBIC)

            calo = np.asarray(calo, dtype=np.float32)
            calo = torch.from_numpy(calo)

            if labels.shape[0]:
                repeat = False
            else:
                index += 1

        return calo, labels

    def __len__(self):
        return self.dataset_size
