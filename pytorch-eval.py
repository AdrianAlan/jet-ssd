from __future__ import print_function

from ssd.generator import CalorimeterJetDataset
from ssd.net import build_ssd
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import sys

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def test_net(model, dataset, top_k, im_size=(300,300),
             conf_threshold=0.05, overlap_threshold=0.1):

    results = np.empty((0,3))
    results2 = np.empty((0,3))

    with torch.no_grad():
        for data, targets in dataset:
            data = data.cuda()
            detections = model(data).data

            targets = targets[0].cpu().numpy()
            targets[:, 0] *= im_size[0]
            targets[:, 2] *= im_size[0]
            targets[:, 1] *= im_size[1]
            targets[:, 3] *= im_size[1]

            agg_detections = np.empty((0,6))
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]

                # Filter detections above given threshold
                dets = dets[dets[:,0] > conf_threshold]

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
                                       labels[:,np.newaxis],
                                       scores[:, np.newaxis])).astype(np.float32, copy=False)
                
                agg_detections = np.vstack((agg_detections, class_det))

            # Sort by confidence
            agg_detections = agg_detections[(-agg_detections[:,-1]).argsort()]

            # Select top k predictions
            agg_detections = agg_detections[:top_k]

            # Loop over agg_detections
            for d in agg_detections:

                detected = 0 
                # Check the overlap
                ixmin = np.maximum(targets[:, 0], d[0])
                iymin = np.maximum(targets[:, 1], d[1])
                ixmax = np.minimum(targets[:, 2], d[2])
                iymax = np.minimum(targets[:, 3], d[3])

                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)

                inters = iw * ih
                uni = ((d[2] - d[0]) * (d[3] - d[1]) +
                       (targets[:,2] - targets[:,0]) * (targets[:,3] - targets[:,1]) - inters) + 10e-12

                overlaps = inters / uni
                overlaps = np.max(overlaps)

                if overlaps > overlap_threshold:
                   detected = 1

                prediction = np.array([d[-1], detected, int(not detected)])
                results = np.vstack((results, prediction))

            # Loop over agg_detections
            for t in targets:

                detected = 0
                # Check the overlap
                ixmin = np.maximum(agg_detections[:, 0], t[0])
                iymin = np.maximum(agg_detections[:, 1], t[1])
                ixmax = np.minimum(agg_detections[:, 2], t[2])
                iymax = np.minimum(agg_detections[:, 3], t[3])

                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)

                inters = iw * ih
                uni = ((t[2] - t[0]) * (t[3] - t[1]) +
                       (agg_detections[:,2] - agg_detections[:,0]) *
                       (agg_detections[:,3] - agg_detections[:,1]) - inters) + 10e-12

                overlaps = inters / uni
                overlaps_x = overlaps.argmax()
                overlaps = np.max(overlaps)

                if overlaps > overlap_threshold:
                   detected = 1

                prediction = np.array([agg_detections[overlaps_x, -1], detected, int(not detected)])
                results2 = np.vstack((results2, prediction))

        results = results[(-results[:,0]).argsort()]
        fp = np.cumsum(results[:,2])
        tp = np.cumsum(results[:,1])
        prec = tp / (tp + fp)

        results2 = results2[(-results2[:,0]).argsort()]
        fn = np.cumsum(results2[:,2])
        tp = np.cumsum(results2[:,1])
        rec = tp / (tp + fn)
        print(rec)
        print(prec)
        ap = voc_ap(rec, prec)
        return ap


if __name__ == '__main__':

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    model_source_path = './models/ssd-jet-2.pth'

    num_classes = 1
    num_classes = num_classes + 1 # +1 for background

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

    top_k = 8
    confidence_threshold = 0.01
    overlap_threshold = 0.25
    img_size = (300, 300)

    for i in range(1, num_classes):
        ap = test_net(net, train_loader, top_k, img_size, confidence_threshold, overlap_threshold)
        print('Average precision for class %s: %s' % (i, ap))
