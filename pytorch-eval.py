from __future__ import print_function

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import sys

from sklearn.metrics import average_precision_score
from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
from time import time
from tqdm import tqdm


def test_net(model, dataset, top_k, im_size=(300, 300),
             conf_threshold=0.05, overlap_threshold=0.1):

    results = np.empty((0, 2))
    inf_time = []

    with torch.no_grad():

        progress_bar = tqdm(total=len(dataset), desc='Evaluating events')

        for data, targets in dataset:
            data = data.cuda()

            t_start = time()

            detections = model(data).data

            t_end = time()
            inf_time.append(t_end-t_start)

            targets = targets[0].cpu().numpy()
            targets[:, 0] *= im_size[0]
            targets[:, 2] *= im_size[0]
            targets[:, 1] *= im_size[1]
            targets[:, 3] *= im_size[1]

            all_detections = np.empty((0, 7))
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]

                # Filter detections above given threshold
                dets = dets[dets[:, 0] > conf_threshold]

                if dets.size(0) == 0:
                    continue

                boxes = dets[:, 1:]
                boxes[:, 0] *= im_size[0]
                boxes[:, 2] *= im_size[0]
                boxes[:, 1] *= im_size[1]
                boxes[:, 3] *= im_size[1]

                scores = dets[:, 0].cpu().numpy()
                labels = np.array([j]*len(scores))

                class_det = np.hstack((boxes.cpu().numpy(),
                                       labels[:, np.newaxis],
                                       scores[:, np.newaxis],
                                       np.zeros(len(boxes))[:, np.newaxis])
                                      ).astype(np.float32, copy=False)

                all_detections = np.vstack((all_detections, class_det))

            # Sort by confidence
            all_detections = all_detections[(-all_detections[:, -1]).argsort()]

            # Select top k predictions
            all_detections = all_detections[:top_k]

            # Detection format: [xmin, ymin, xmax, ymax, label, score, tp/fp]
            # Loop over detections
            for t in targets:
                detected = False

                for x, d in enumerate(all_detections):
                    ixmin = np.maximum(t[0], d[0])
                    iymin = np.maximum(t[1], d[1])
                    ixmax = np.minimum(t[2], d[2])
                    iymax = np.minimum(t[3], d[3])

                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    intersection = iw * ih

                    union = ((d[2] - d[0]) * (d[3] - d[1]) +
                             (t[2] - t[0]) * (t[3] - t[1]) - intersection)

                    overlap = intersection / (union + 10e-12)

                    if overlap > overlap_threshold:
                        detected = True
                        all_detections[x][6] = 1
                        break

                if detected:
                    fn = np.hstack((np.zeros(6), [1])).astype(np.float32,
                                                              copy=False)
                    all_detections = np.vstack((all_detections, fn))

                results = np.vstack((results, all_detections[:, [6, 5]]))

            progress_bar.update(1)

        progress_bar.close()

        return (round(average_precision_score(results[:, 0],
                                              results[:, 1]), 2),
                round(1000*np.mean(inf_time)))


if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model_source_path = './models/ssd-jet-tests.pth'

    num_classes = 1
    num_classes = num_classes + 1  # +1 for background

    net = build_ssd('test', 300, num_classes, False)
    net.load_weights(model_source_path)
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True
    print('Finished loading model')

    train_dataset_path = '/eos/user/a/adpol/ceva/fast/RSGraviton_NARROW_0.h5'
    h5_train = h5py.File(train_dataset_path, 'r')
    train_dataset = CalorimeterJetDataset(hdf5_dataset=h5_train)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1)

    for i in range(1, num_classes):
        (ap, it) = test_net(net, train_loader, top_k=10, im_size=(300, 300),
                            conf_threshold=0.01, overlap_threshold=0.5)
        print('Average precision for class %s: %s' % (i, ap))
        print('Average inference time for class %s: %s ms' % (i, it))
