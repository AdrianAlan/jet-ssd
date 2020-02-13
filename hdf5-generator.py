#!/usr/bin/env python

import argparse
import h5py
import numpy as np
import os
import simplejson as json
import uproot
import warnings

from collections import Counter
from tqdm import tqdm


class PhysicsConstants():

    def __init__(self, example_file):

        # Define quarks mass, in GeV
        self.q_mass = {'b': 4.18, 'q': 0.096, 't': 173.1, 'W': 80.39,
                       'h': 124.97}

        # Define radius of a jet
        self.delta_r = {'b': 0.4, 'q': 0.4, 't': 0.8, 'W': 0.8, 'h': 0.8}

        # Crystal dimentions
        self.crystal_dim = 0.0174

        # Assigned jet classes
        self.classes = {'b': 1, 'q': 5, 't': 4, 'W': 3, 'h': 2}

        # Offset in phi
        self.offset_phi = self.get_radius_in_pixels('t')

        self.example_file = example_file
        self.min_eta = -3
        self.max_eta = 3

    def get_class(self, jtype):
        return self.classes[jtype]

    # Width of Jets,
    def get_radius_in_pixels(self, jtype):
        return np.ceil(self.delta_r[jtype] / self.crystal_dim).astype(int)

    # Minimum detecable Pt
    def get_minimum_pt(self, jtype):
        return 2*self.q_mass[jtype] / self.delta_r[jtype]

    def get_edges_ecal(self, edge_index, tower, sample_events=1000):
        file = uproot.open(self.example_file)
        tower_flag_full_file = file['Delphes']['Tower'][tower].array()
        edges_full_file = file['Delphes']['Tower']['Tower.Edges[4]'].array()

        global_edges = np.array([], dtype=np.float32)

        for i in range(sample_events):

            edges_event = edges_full_file[i][tower_flag_full_file[i] > 0]
            global_edges = np.append(global_edges,
                                     edges_event[:, edge_index])
            global_edges = np.append(global_edges,
                                     edges_event[:, edge_index+1])
            global_edges = np.unique(global_edges)

        if edge_index == 0:
            global_edges = global_edges[(global_edges > self.min_eta) &
                                        (global_edges < self.max_eta)]

        return global_edges


class HDF5Generator:

    def __init__(self,
                 jtype,
                 hdf5_dataset_path,
                 hdf5_dataset_size,
                 example_file,
                 files_details):

        self.constants = PhysicsConstants(example_file)

        self.jtype = jtype
        self.radius = self.constants.get_radius_in_pixels(jtype)
        self.minpt = self.constants.get_minimum_pt(jtype)
        self.edges_eta_ecal = self.constants.get_edges_ecal(edge_index=0,
                                                            tower='Tower.Eem')
        self.edges_phi_ecal = self.constants.get_edges_ecal(edge_index=2,
                                                            tower='Tower.Eem')
        self.edges_eta_hcal = self.constants.get_edges_ecal(edge_index=0,
                                                            tower='Tower.Ehad')
        self.edges_phi_hcal = self.constants.get_edges_ecal(edge_index=2,
                                                            tower='Tower.Ehad')
        self.unique_label = self.constants.get_class(jtype)
        self.hdf5_dataset_path = hdf5_dataset_path
        self.hdf5_dataset_size = hdf5_dataset_size
        self.files_details = files_details

    def create_hdf5_dataset(self, progress_bar):

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'w')

        # Create dataset where the images will be stored as flattened arrays
        hdf5_calorimeter = hdf5_dataset.create_dataset(
                name='calorimeter',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.uint16))

        # Create dataset where the labels are stored as flattened arrays
        hdf5_labels = hdf5_dataset.create_dataset(
                name='labels',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.uint16))

        # Create dataset that will hold the dimensions of labels arrays
        hdf5_label_shapes = hdf5_dataset.create_dataset(
                name='label_shapes',
                shape=(self.hdf5_dataset_size, 2),
                maxshape=(None, 2),
                dtype=np.uint8)

        i = 0

        for file_details in self.files_details:
            file_path = next(iter(file_details.keys()))

            events = file_details[file_path]

            # First load values from file
            file = uproot.open(file_path)

            towers = file['Delphes']['Tower']
            ecal_energy_full_file = towers['Tower.Eem'].array()  # ECAL E
            hcal_energy_full_file = towers['Tower.Ehad'].array()  # HCAL E
            edges_full_file = towers['Tower.Edges[4]'].array()  # Crystal edge

            genjet = file['Delphes']['GenJet']
            bbox_pt_full_file = genjet['GenJet.PT'].array()  # Jet PT
            bbox_eta_full_file = genjet['GenJet.Eta'].array()  # Jet x
            bbox_phi_full_file = genjet['GenJet.Phi'].array()  # Jet y

            for event_number in np.arange(events[0], events[1], dtype=int):

                # Get jet labels
                bbox_etas = bbox_eta_full_file[event_number]
                bbox_phis = bbox_phi_full_file[event_number]

                etas_mask = ((bbox_etas > self.edges_eta_ecal[0]) &
                             (bbox_etas < self.edges_eta_ecal[-1]))

                bbox_etas = bbox_etas[etas_mask]
                bbox_phis = bbox_phis[etas_mask]
                bbox_pts = bbox_pt_full_file[event_number][etas_mask]

                labels = self.get_bboxes(bbox_pts, bbox_etas, bbox_phis)

                # Get ECAL info
                ecal_mask = ecal_energy_full_file[event_number] > 0

                etas = edges_full_file[event_number][ecal_mask][:, 0]
                phis = edges_full_file[event_number][ecal_mask][:, 2]
                energy = ecal_energy_full_file[event_number][ecal_mask]

                etas_mask = ((etas > self.edges_eta_ecal[0]) &
                             (etas < self.edges_eta_ecal[-1]))

                etas = etas[etas_mask]
                phis = phis[etas_mask]
                energy = energy[etas_mask]

                pixels_ecal = self.get_energy_map(etas,
                                                  phis,
                                                  energy,
                                                  cal='ecal')

                # Get HCAL info
                hcal_mask = hcal_energy_full_file[event_number] > 0

                etas = edges_full_file[event_number][hcal_mask][:, 0]
                phis = edges_full_file[event_number][hcal_mask][:, 2]
                energy = hcal_energy_full_file[event_number][hcal_mask]

                etas_mask = ((etas > self.edges_eta_hcal[0]) &
                             (etas < self.edges_eta_hcal[-1]))

                etas = etas[etas_mask]
                phis = phis[etas_mask]
                energy = energy[etas_mask]

                pixels_hcal = self.get_energy_map(etas,
                                                  phis,
                                                  energy,
                                                  cal='hcal')

                # Push the data to hdf5
                # Flatten the image array and write it to the images dataset.
                calorimeter = np.append(pixels_ecal, pixels_hcal)

                hdf5_calorimeter[i] = calorimeter

                # Flatten the labels array and write it to the labels dataset.
                hdf5_labels[i] = labels.reshape(-1)

                # Write the labels' shape to the label shapes dataset.
                hdf5_label_shapes[i] = labels.shape

                i = i + 1

                progress_bar.update(1)

        hdf5_dataset.close()

    def get_bboxes(self, bbox_pts, bbox_etas, bbox_phis):

        labels = []

        for pt, eta, phi in zip(bbox_pts, bbox_etas, bbox_phis):

            if pt > self.minpt:
                index_phi = np.argmax(self.edges_phi_ecal >= phi) - 1
                index_eta = np.argmax(self.edges_eta_ecal >= eta) - 1

                index_phi = index_phi + self.constants.offset_phi

                xmin = int(index_eta-self.radius)
                xmax = int(index_eta+self.radius)
                ymin = int(index_phi-self.radius)
                ymax = int(index_phi+self.radius)

                if xmin < 0:
                    xmin = 0
                if xmax > len(self.edges_eta_ecal) - 1:
                    xmax = len(self.edges_eta_ecal) - 1

                labels.append([self.unique_label, xmin, ymin, xmax, ymax, pt])

        return np.asarray(labels)

    def get_energy_map(self, etas, phis, energy, cal='ecal'):

        coordinates = []

        for eta, phi in zip(etas, phis):

            if cal == 'ecal':
                index_phi = np.where(self.edges_phi_ecal == phi)[0][0]
                index_eta = np.where(self.edges_eta_ecal == eta)[0][0]
            else:
                index_phi = np.where(self.edges_phi_hcal == phi)[0][0]
                index_eta = np.where(self.edges_eta_hcal == eta)[0][0]

            coordinates.append((index_phi, index_eta))

        if cal == 'ecal':
            pixels = np.zeros((len(self.edges_phi_ecal)-1,
                               len(self.edges_eta_ecal)-1))
        else:
            pixels = np.zeros((len(self.edges_phi_hcal)-1,
                               len(self.edges_eta_hcal)-1))

        for c, e in zip(coordinates, energy):
            pixels[c[0]][c[1]] = e

        if cal != 'ecal':
            pixels = np.hstack((pixels[:, :85],
                                pixels[:, 85:-85].repeat(5, axis=1),
                                pixels[:, -85:]))

        # Extend image top and bottom by a radius of the biggest jet
        # Prevents cropped boxes in y plane
        pixels = np.vstack((pixels[-self.constants.offset_phi:, :],
                            pixels,
                            pixels[:self.constants.offset_phi, :]))

        return pixels


class Utils():

    def parse_config(self, jtype, nofiles, config_path, source_files_dir):

        # Laod configuration
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        # Total number of events
        total = config[jtype]['events']
        files_details = []
        files_batch = []
        file_id = 0
        batch_id = 1
        batch_size = total / float(nofiles)
        gtotal = 0
        event_min_next = 0

        while gtotal < total:
            # Missing file skip
            if str(file_id) not in config[jtype]['files'].keys():
                file_id = file_id + 1
                continue

            # Set FROM and TO indexes
            event_min = event_min_next
            event_max = config[jtype]['files'][str(file_id)]

            # Set file name
            ftype = 'RSGraviton_%s_NARROW' % jtype
            file = "%s/%s/%s_%s.root" % (source_files_dir,
                                         ftype, ftype, file_id)

            # Fix nominal target of events
            gtotal_target = gtotal + event_max - event_min

            # Save filenames with indexes

            # When event_max has to be cropped
            if batch_id*batch_size <= gtotal_target:
                max_in_this_batch = int(batch_id*batch_size)
                event_max = event_max - (gtotal_target - max_in_this_batch)
                event_min_next = event_max

                # Prevent saving files with no events
                if event_max != event_min:
                    files_batch.append({file: (event_min, event_max)})

                # Push to file details
                files_details.append(files_batch)
                files_batch = []
                batch_id = batch_id + 1
            # Full file
            else:
                files_batch.append({file: (event_min, event_max)})
                event_min_next = 0
                file_id = file_id + 1

            gtotal = gtotal + event_max - event_min

        return files_details, batch_size, gtotal


class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid path'.format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                    '{0} is not a readable directory'.format(prospective_dir))


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert root file data to h5')

    parser.add_argument('jtype',
                        type=str,
                        choices={'bb', 'tt', 'WW', 'hh'},
                        help='Type of jet to convert')

    parser.add_argument('-n', '--number-of-files',
                        type=int,
                        default=10,
                        help='Target number of output files',
                        dest='nfiles')

    parser.add_argument('-o', '--save-path',
                        type=str,
                        action=IsReadableDir,
                        default='.',
                        help='Output directory',
                        dest='savedir')

    parser.add_argument('-s', '--source-path',
                        type=str,
                        action=IsReadableDir,
                        default='/eos/project/d/dshep/CEVA',
                        help='Source directory',
                        dest='sourcedir')

    parser.add_argument('-c', '--configuration',
                        type=str,
                        action=IsReadableDir,
                        default='./data/file-configuration.json',
                        help='File configuration path',
                        dest='configuration')

    args = parser.parse_args()

    utils = Utils()

    files_details, batch_size, total_events = utils.parse_config(
        args.jtype,
        args.nfiles,
        args.configuration,
        args.sourcedir)

    jet_type = args.jtype[0]

    progress_bar = tqdm(total=total_events,
                        desc=('Creating HDF5 Dataset for %s jets' % jet_type))

    for index, file_dict in enumerate(files_details):

        if index > 1:
            continue

        dataset_size = int((index+1)*batch_size)-int((index)*batch_size)

        generator = HDF5Generator(
                jtype=jet_type,
                hdf5_dataset_path='%s/RSGraviton_%s_NARROW_%s-full.h5' %
                                  (args.savedir, args.jtype, index),
                hdf5_dataset_size=dataset_size,
                example_file='%s/RSGraviton_%s/RSGraviton_%s_0.root' %
                             (args.sourcedir, 'WW_NARROW', 'WW_NARROW'),
                files_details=file_dict)
        generator.create_hdf5_dataset(progress_bar)

    progress_bar.close()
