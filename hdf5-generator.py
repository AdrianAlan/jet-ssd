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

    def __init__(self):

        # Define quarks mass, in GeV
        self.q_mass = {'b': 4.18, 'q': 0.096, 't': 173.1, 'W': 80.39,
                       'h': 124.97}

        # Define radius of a jet
        self.delta_r = {'b': 0.4, 'q': 0.4, 't': 0.8, 'W': 0.8, 'h': 0.8}

        # Crystal dimentions
        self.crystal_dim = 0.0174

        # Assigned jet classes
        self.classes = {'b': 1, 'q': 5, 't': 4, 'W': 3, 'h': 2}

        # Hardcoded Pi (legacy)
        self.pi = np.float128(3.141592653589793238462643383279502884197)

        # Offset in phi
        self.offset_phi = self.get_radius_in_pixels('t')

    def get_class(self, jtype):
        return self.classes[jtype]

    # Width of Jets,
    def get_radius_in_pixels(self, jtype):
        return np.ceil(self.delta_r[jtype] / self.crystal_dim).astype(int)

    # Minimum detecable Pt
    def get_minimum_pt(self, jtype):
        return 2*self.q_mass[jtype] / self.delta_r[jtype]

    # Grid edges in phi and eta
    def get_edges_phi(self):
        phi = []
        for i in np.arange(-180, 180):
            phi.append(i * self.pi/180.0)
        return np.array(phi).astype(np.float32)

    def get_edges_eta(self):
        # Scale is used to ensure precision
        scale = 10000.0
        eta = []
        for i in np.arange(-2.958*scale,
                           2.958*scale,
                           self.crystal_dim*scale,
                           dtype=int):
            eta.append(float(i)/scale)
        return np.array(eta).astype(np.float32)


class HDF5Generator:

    def __init__(self,
                 jtype,
                 hdf5_dataset_path,
                 hdf5_dataset_size,
                 files_details):

        self.constants = PhysicsConstants()

        self.jtype = jtype
        self.radius = self.constants.get_radius_in_pixels(jtype)
        self.minpt = self.constants.get_minimum_pt(jtype)
        self.edges_phi = self.constants.get_edges_phi()
        self.edges_eta = self.constants.get_edges_eta()
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
                dtype=h5py.special_dtype(vlen=np.uint8))

        # Create dataset where the labels are stored as flattened arrays
        hdf5_labels = hdf5_dataset.create_dataset(
                name='labels',
                shape=(self.hdf5_dataset_size,),
                maxshape=(None),
                dtype=h5py.special_dtype(vlen=np.int32))

        # Create dataset that will hold the dimensions of labels arrays
        hdf5_label_shapes = hdf5_dataset.create_dataset(
                name='label_shapes',
                shape=(self.hdf5_dataset_size, 2),
                maxshape=(None, 2),
                dtype=np.int32)

        i = 0

        for file_details in self.files_details:
            file_path = next(iter(file_details.keys()))

            events = file_details[file_path]

            # First load values from file
            file = uproot.open(file_path)

            towers = file['Delphes']['Tower']
            phis_full_file = towers['Tower.Phi'].array()
            etas_full_file = towers['Tower.Eta'].array()
            energy_full_file = towers['Tower.E'].array()
            is_ecal_full_file = towers['Tower.Eem'].array()
            # is_hcal_full_file = towers['Tower.Ehad'].array()

            genjet = file['Delphes']['GenJet']
            bbox_pt_full_file = genjet['GenJet.PT'].array()
            bbox_eta_full_file = genjet['GenJet.Eta'].array()
            bbox_phi_full_file = genjet['GenJet.Phi'].array()

            for event_number in np.arange(events[0], events[1], dtype=int):

                # Get ECAL mask
                ecal_mask = is_ecal_full_file[event_number] > 0

                # Load values for one event
                phis = phis_full_file[event_number][ecal_mask]
                etas = etas_full_file[event_number][ecal_mask]
                energy = energy_full_file[event_number][ecal_mask]

                bbox_pts = bbox_pt_full_file[event_number]
                bbox_etas = bbox_eta_full_file[event_number]
                bbox_phis = bbox_phi_full_file[event_number]

                # Get pixel intensities
                pixels = self.get_energy_map(etas, phis, energy)

                # Get labels
                labels = self.get_bboxes(bbox_pts, bbox_etas, bbox_phis)

                # Push the data to hdf5
                # Flatten the image array and write it to the images dataset.
                hdf5_calorimeter[i] = pixels.reshape(-1)

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
                index_phi = np.argmax(self.edges_phi >= phi) - 1
                index_eta = np.argmax(self.edges_eta >= eta) - 1

                # Correct for eta > edges_eta[-1]

                index_phi = index_phi + self.constants.offset_phi

                xmin = int(index_eta-self.radius)
                xmax = int(index_eta+self.radius)
                ymin = int(index_phi-self.radius)
                ymax = int(index_phi+self.radius)

                if xmin < 0:
                    xmin = 0

                if xmax > 339:
                    xmax = 339

                labels.append([self.unique_label, xmin, ymin, xmax, ymax])

        return np.asarray(labels)

    def get_energy_map(self, etas, phis, energy):

        # Filter only center of the calorimeter

        eta_mask = (etas > self.edges_eta[0]) & (etas < self.edges_eta[-1])
        phis = phis[eta_mask].astype(np.float32)
        etas = etas[eta_mask].astype(np.float32)
        energy = energy[eta_mask]

        coordinates = []

        for eta, phi in zip(etas, phis):

            index_phi = np.argmax(self.edges_phi >= phi) - 1
            index_eta = np.argmax(self.edges_eta >= eta) - 1

            # Sanity check
            # Two crystals in one px because one lands on the edge
            coords = (index_phi, index_eta)
            if coords in coordinates:
                if eta in self.edges_eta:
                    index_eta = index_eta + 1
                if phi in self.edges_phi:
                    index_phi = index_phi + 1

            coordinates.append((index_phi, index_eta))

        # Sanity check if unique set of coordinates
        if not len(list(set(coordinates))) == len(coordinates):
            # Just for debugging
            coo = [i for i, c in Counter(coordinates).items() if c > 1][0]
            x1 = coordinates.index(coo)
            x2 = coordinates[x1+1:].index(coo)+1+x1
            warnings.warn("Multiple crystals per pixel for coordinates %s." +
                          " Affected Towers: %s %s and %s %s" % (coo,
                                                                 phis[x1],
                                                                 etas[x1],
                                                                 phis[x2],
                                                                 etas[x2]))

        pixels = np.zeros((360, 340))
        for c, e in zip(coordinates, energy):
            pixels[c[0]][c[1]] = e

        # Extend image top and bottom by a radius of the biggest jet
        # Prevents cropped boxes in y plane
        pixels = np.vstack((pixels[-self.constants.offset_phi:, :],
                            pixels,
                            pixels[:self.constants.offset_phi, :]))

        # Make it 3 dimensional
        pixels = np.expand_dims(pixels, axis=2)

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
                        default='./',
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
                        default='data/file-configuration.json',
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
        dataset_size = int((index+1)*batch_size)-int((index)*batch_size)
        generator = HDF5Generator(
                jtype=jet_type,
                hdf5_dataset_path='%sRSGraviton_%s_NARROW_%s.h5' %
                                  (args.savedir, args.jtype, index),
                hdf5_dataset_size=dataset_size,
                files_details=file_dict)
        generator.create_hdf5_dataset(progress_bar)

    progress_bar.close()
