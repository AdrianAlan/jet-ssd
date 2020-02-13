from __future__ import division
import numpy as np
from math import ceil
from tqdm import trange
import sys
import warnings

from ssd.generator import DataGenerator
from ssd.ssd_output_decoder import decode_detections

from ssd.bounding_box_utils import iou

class Evaluator:
    '''Computes the mean average precision of the given Keras SSD model on the given dataset.'''

    def __init__(self,
                 model,
                 n_classes,
                 data_generator,
                 model_mode='inference',
                 pred_format={'class_id': 0, 'conf': 1, 'xmin': 2, 'ymin': 3, 'xmax': 4, 'ymax': 5},
                 gt_format={'class_id': 0, 'xmin': 1, 'ymin': 2, 'xmax': 3, 'ymax': 4}):

        if not isinstance(data_generator, DataGenerator):
            warnings.warn("`data_generator` is not a `DataGenerator` object, which will cause undefined behavior.")

        self.model = model
        self.data_generator = data_generator
        self.n_classes = n_classes
        self.model_mode = model_mode
        self.pred_format = pred_format
        self.gt_format = gt_format

        # The following lists all contain per-class data, i.e. all list have the length `n_classes + 1`, where one element is for the background class,
        self.prediction_results = None
        self.num_gt_per_class = None
        self.true_positives = None
        self.false_positives = None
        self.cumulative_true_positives = None
        self.cumulative_false_positives = None
        self.cumulative_precisions = None
        self.cumulative_recalls = None
        self.average_precisions = None
        self.mean_average_precision = None
        self.ground_truth_labels = None
        self.image_ids = None

    def __call__(self,
                 img_height,
                 img_width,
                 batch_size,
                 round_confidences=False,
                 matching_iou_threshold=0.5,
                 border_pixels='include',
                 sorting_algorithm='quicksort',
                 average_precision_mode='integrate',
                 num_recall_points=11,
                 ignore_neutral_boxes=True,
                 verbose=True,
                 decoding_confidence_thresh=0.01,
                 decoding_iou_threshold=0.45,
                 decoding_top_k=200,
                 decoding_pred_coords='centroids',
                 decoding_normalize_coords=True):
        '''
        Computes the mean average precision of the given Keras SSD model on the given dataset.

        Optionally also returns the averages precisions, precisions, and recalls.

        All the individual steps of the overall evaluation algorithm can also be called separately
        (check out the other methods of this class), but this runs the overall algorithm all at once.

        Arguments:
            img_height (int): The input image height for the model.
            img_width (int): The input image width for the model.
            batch_size (int): The batch size for the evaluation.
            round_confidences (int, optional): `False` or an integer that is the number of decimals that the prediction
                confidences will be rounded to. If `False`, the confidences will not be rounded.
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            average_precision_mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision
                will be computed according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled
                for `num_recall_points` recall values. In the case of 'integrate', the average precision will be computed according to the
                Pascal VOC formula that was used from VOC 2010 onward, where the average precision will be computed by numerically integrating
                over the whole preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just
                the limit case of 'sample' mode as the number of sample points increases.
            num_recall_points (int, optional): The number of points to sample from the precision-recall-curve to compute the average
                precisions. In other words, this is the number of equidistant recall values for which the resulting precision will be
                computed. 11 points is the value used in the official Pascal VOC 2007 detection evaluation algorithm.
            ignore_neutral_boxes (bool, optional): In case the data generator provides annotations indicating whether a ground truth
                bounding box is supposed to either count or be neutral for the evaluation, this argument decides what to do with these
                annotations. If `False`, even boxes that are annotated as neutral will be counted into the evaluation. If `True`,
                neutral boxes will be ignored for the evaluation. An example for evaluation-neutrality are the ground truth boxes
                annotated as "difficult" in the Pascal VOC datasets, which are usually treated as neutral for the evaluation.
            return_precisions (bool, optional): If `True`, returns a nested list containing the cumulative precisions for each class.
            return_recalls (bool, optional): If `True`, returns a nested list containing the cumulative recalls for each class.
            return_average_precisions (bool, optional): If `True`, returns a list containing the average precision for each class.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            decoding_confidence_thresh (float, optional): Only relevant if the model is in 'training' mode.
                A float in [0,1), the minimum classification confidence in a specific positive class in order to be considered
                for the non-maximum suppression stage for the respective class. A lower value will result in a larger part of the
                selection process being done by the non-maximum suppression stage, while a larger value will result in a larger
                part of the selection process happening in the confidence thresholding stage.
            decoding_iou_threshold (float, optional): Only relevant if the model is in 'training' mode. A float in [0,1].
                All boxes with a Jaccard similarity of greater than `iou_threshold` with a locally maximal box will be removed
                from the set of predictions for a given class, where 'maximal' refers to the box score.
            decoding_top_k (int, optional): Only relevant if the model is in 'training' mode. The number of highest scoring
                predictions to be kept for each batch item after the non-maximum suppression stage.
            decoding_input_coords (str, optional): Only relevant if the model is in 'training' mode. The box coordinate format
                that the model outputs. Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or 'corners' for the format `(xmin, ymin, xmax, ymax)`.
            decoding_normalize_coords (bool, optional): Only relevant if the model is in 'training' mode. Set to `True` if the model
                outputs relative coordinates. Do not set this to `True` if the model already outputs absolute coordinates,
                as that would result in incorrect coordinates.

        Returns:
            A float, the mean average precision, plus any optional returns specified in the arguments.
        '''

        # Predict on the entire dataset.
        self.predict_on_dataset(batch_size=batch_size, round_confidences=round_confidences)

        # Get the total number of ground truth boxes for each class.
        self.get_num_gt_per_class()

        # Match predictions to ground truth boxes for all classes.
        self.match_predictions(matching_iou_threshold=matching_iou_threshold,
                               border_pixels=border_pixels,
                               sorting_algorithm=sorting_algorithm)

        # Compute the cumulative precision and recall for all classes.
        self.compute_precision_recall()

        # Compute the average precision for this class.
        self.compute_average_precisions(mode=average_precision_mode,
                                        num_recall_points=num_recall_points)

        # Compute the mean average precision.
        self.compute_mean_average_precision()

        return self.mean_average_precision, self.average_precisions, self.cumulative_precisions, self.cumulative_recalls

    def predict_on_dataset(self, batch_size, round_confidences=False):
        '''Runs predictions for the given model over the entire dataset given by `data_generator`.'''

        class_id_pred = self.pred_format['class_id']
        conf_pred     = self.pred_format['conf']
        xmin_pred     = self.pred_format['xmin']
        ymin_pred     = self.pred_format['ymin']
        xmax_pred     = self.pred_format['xmax']
        ymax_pred     = self.pred_format['ymax']
        
        class_id_gt = self.gt_format['class_id']
        xmin_gt = self.gt_format['xmin']
        ymin_gt = self.gt_format['ymin']
        xmax_gt = self.gt_format['xmax']
        ymax_gt = self.gt_format['ymax']

        # Configure the data generator for the evaluation.
        generator = self.data_generator.generate(batch_size=batch_size,
                                                 shuffle=False,
                                                 label_encoder=None,
                                                 returns={'image_ids',
                                                          'processed_images',
                                                          'processed_labels'})

        #############################################################################################
        # Predict over all batches of the dataset and store the predictions.
        #############################################################################################

        # Compute the number of batches to iterate over the entire dataset.
        n_images = self.data_generator.get_dataset_size()
        n_batches = int(ceil(n_images / batch_size))
        
        # We have to generate a separate results list for each class.
        results = [list() for _ in range(self.n_classes + 1)]
        labels = [list() for _ in range(n_images)]
        
        print("Number of histograms in the evaluation dataset: {}".format(n_images))
        tr = trange(n_batches, file=sys.stdout)
        tr.set_description('Generating predictions')

        # Loop over all batches.
        for j in tr:
            
            # Generate batch.
            batch_image_ids, batch_X, batch_y = next(generator)
            
            # Predict.
            y_pred = self.model.predict(batch_X)
            
            # Filter out the all-zeros dummy elements of `y_pred`.
            y_pred_filtered = []
            for i in range(len(y_pred)):
                y_pred_filtered.append(y_pred[i][y_pred[i,:,0] != 0])
            y_pred = y_pred_filtered

            # Iterate over all batch items.
            for k, (batch_pred, batch_true) in enumerate(zip(y_pred, batch_y)):

                image_id = batch_image_ids[k]

                for box in batch_pred:
                    class_id = int(box[class_id_pred])
                    
                    # Round the box coordinates to reduce the required memory.
                    if round_confidences:
                        confidence = round(box[conf_pred], round_confidences)
                    else:
                        confidence = box[conf_pred]
                    
                    xmin = round(box[xmin_pred], 1)
                    ymin = round(box[ymin_pred], 1)
                    xmax = round(box[xmax_pred], 1)
                    ymax = round(box[ymax_pred], 1)
                    prediction = (image_id, confidence, xmin, ymin, xmax, ymax)
                    # Append the predicted box to the results list for its class.
                    results[class_id].append(prediction)

                for box in batch_true:
                    class_id = int(box[class_id_gt])
                    xmin = round(box[xmin_gt], 1)
                    ymin = round(box[ymin_gt], 1)
                    xmax = round(box[xmax_gt], 1)
                    ymax = round(box[ymax_gt], 1)
                    label = (class_id, xmin, ymin, xmax, ymax)
                    labels[image_id].append(label)

        self.ground_truth_labels = labels
        self.prediction_results = results #image_id

    def get_num_gt_per_class(self):
        '''Counts the number of ground truth boxes for each class across the dataset.'''

        print('Computing the number of positive ground truth boxes per class.')
        
        if self.ground_truth_labels is None:
            raise ValueError('Computing the number of ground truth boxes per class not possible, no ground truth given.')

        num_gt_per_class = np.zeros(shape=(self.n_classes+1), dtype=np.int)
        class_id_gt = self.gt_format['class_id']
        
        for labels in self.ground_truth_labels:
            for label in labels:
                class_id = label[class_id_gt]
                num_gt_per_class[class_id] += 1

        self.num_gt_per_class = num_gt_per_class
        
    def match_predictions(self,
                          matching_iou_threshold=0.5,
                          border_pixels='include',
                          sorting_algorithm='quicksort'):
        '''Matches predictions to ground truth boxes.

        Arguments:
            matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a Jaccard overlap
                of at least `matching_iou_threshold` with any ground truth bounding box of the same class.
            border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
                Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
                to the boxes. If 'exclude', the border pixels do not belong to the boxes.
                If 'half', then one of each of the two horizontal and vertical borders belong
                to the boxex, but not the other.
            sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument accepts
                any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose between 'quicksort'
                (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower and less memory efficient, but stable).
                The official Matlab evaluation algorithm uses a stable sorting algorithm, so this algorithm is only guaranteed
                to behave identically if you choose 'mergesort' as the sorting algorithm, but it will almost always behave identically
                even if you choose 'quicksort' (but no guarantees).
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the true and false positives.

        Returns:
            None by default. Optionally, four nested lists containing the true positives, false positives, cumulative true positives,
            and cumulative false positives for each class.
        '''

        if self.ground_truth_labels is None:
            raise ValueError("Matching predictions to ground truth boxes not possible, no ground truth given.")

        if self.prediction_results is None:
            raise ValueError("There are no prediction results. You must run `predict_on_dataset()` before calling this method.")

        class_id_gt = self.gt_format['class_id']
        xmin_gt = self.gt_format['xmin']
        ymin_gt = self.gt_format['ymin']
        xmax_gt = self.gt_format['xmax']
        ymax_gt = self.gt_format['ymax']
        
        ground_truth = self.ground_truth_labels

        true_positives = [[]] # The false positives for each class, sorted by descending confidence.
        false_positives = [[]] # The true positives for each class, sorted by descending confidence.
        cumulative_true_positives = [[]]
        cumulative_false_positives = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            predictions = self.prediction_results[class_id]

            # Store the matching results in these lists:
            true_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a true positive, 0 otherwise
            false_pos = np.zeros(len(predictions), dtype=np.int) # 1 for every prediction that is a false positive, 0 otherwise

            # In case there are no predictions at all for this class, we're done here.
            if len(predictions) == 0:
                print("No predictions for class {}/{}".format(class_id, self.n_classes))
                true_positives.append(true_pos)
                false_positives.append(false_pos)
                
                cumulative_true_pos = np.cumsum(true_pos) # Cumulative sums of the true positives
                cumulative_false_pos = np.cumsum(false_pos) # Cumulative sums of the false positives

                cumulative_true_positives.append(cumulative_true_pos)
                cumulative_false_positives.append(cumulative_false_pos)
                
                continue

            # Convert the predictions list for this class into a structured array so that we can sort it by confidence.

            # Get the number of characters needed to store the image ID strings in the structured array.
            # Keep a few characters buffer in case some image IDs are longer than others.
            num_chars_per_image_id = len(str(predictions[0][0])) + 6
            
            # Create the data type for the structured array.
            preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_per_image_id)),
                                        ('confidence', 'f4'),
                                        ('xmin', 'f4'),
                                        ('ymin', 'f4'),
                                        ('xmax', 'f4'),
                                        ('ymax', 'f4')])
            
            # Create the structured array
            predictions = np.array(predictions, dtype=preds_data_type)

            # Sort the detections by decreasing confidence.
            descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm)
            predictions_sorted = predictions[descending_indices]

            print("Matching predictions to ground truth, class {}/{}.".format(class_id, self.n_classes))

            # Keep track of which ground truth boxes were already matched to a detection.
            gt_matched = {}

            # Iterate over all predictions.
            for i in range(len(predictions)):

                prediction = predictions_sorted[i]
                image_id = prediction['image_id']
                
                # Convert the structured array element to a regular array.
                pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']])) 

                # Get the relevant ground truth boxes for this prediction,
                # i.e. all ground truth boxes that match the prediction's
                # image ID and class ID.

                # The ground truth could either be a tuple with `(ground_truth_boxes, eval_neutral_boxes)`
                # or only `ground_truth_boxes`.

                # Why do I have to cast it here? I don't know
                image_id = int(image_id)

                gt = self.ground_truth_labels[image_id]
                gt = np.asarray(gt)
              
                
                # Check if not empty
                if gt.size != 0:
                    class_mask = gt[:,class_id_gt] == class_id
                    gt = gt[class_mask]

                if gt.size == 0:
                    # If the image doesn't contain any objects of this class,
                    # the prediction becomes a false positive.
                    false_pos[i] = 1
                    continue

                # Compute the IoU of this prediction with all ground truth boxes of the same class.
                overlaps = iou(boxes1=gt[:,[xmin_gt, ymin_gt, xmax_gt, ymax_gt]],
                               boxes2=pred_box,
                               coords='corners',
                               mode='element-wise',
                               border_pixels=border_pixels)

                # For each detection, match the ground truth box with the highest overlap.
                # It's possible that the same ground truth box will be matched to multiple
                # detections.
                gt_match_index = np.argmax(overlaps)
                gt_match_overlap = overlaps[gt_match_index]

                if gt_match_overlap < matching_iou_threshold:
                    # False positive, IoU threshold violated:
                    # Those predictions whose matched overlap is below the threshold become
                    # false positives.
                    false_pos[i] = 1
                else:
                    if not (image_id in gt_matched):
                        # True positive:
                        # If the matched ground truth box for this prediction hasn't been matched to a
                        # different prediction already, we have a true positive.
                        true_pos[i] = 1
                        gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                        gt_matched[image_id][gt_match_index] = True
                    elif not gt_matched[image_id][gt_match_index]:
                        # True positive:
                        # If the matched ground truth box for this prediction hasn't been matched to a
                        # different prediction already, we have a true positive.
                        true_pos[i] = 1
                        gt_matched[image_id][gt_match_index] = True
                    else:
                        # False positive, duplicate detection:
                        # If the matched ground truth box for this prediction has already been matched
                        # to a different prediction previously, it is a duplicate detection for an
                        # already detected object, which counts as a false positive.
                        false_pos[i] = 1

            true_positives.append(true_pos)
            false_positives.append(false_pos)

            cumulative_true_pos = np.cumsum(true_pos) # Cumulative sums of the true positives
            cumulative_false_pos = np.cumsum(false_pos) # Cumulative sums of the false positives

            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)

        self.true_positives = true_positives
        self.false_positives = false_positives
        self.cumulative_true_positives = cumulative_true_positives
        self.cumulative_false_positives = cumulative_false_positives

    def compute_precision_recall(self):
        '''Computes the precisions and recalls for all classes.'''

        if (self.cumulative_true_positives is None) or (self.cumulative_false_positives is None):
            raise ValueError("True and false positives not available. You must run `match_predictions()` before you call this method.")

        if (self.num_gt_per_class is None):
            raise ValueError("Number of ground truth boxes per class not available. You must run `get_num_gt_per_class()` before you call this method.")

        cumulative_precisions = [[]]
        cumulative_recalls = [[]]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            print("Computing precisions and recalls, class {}/{}".format(class_id, self.n_classes))
            tp = self.cumulative_true_positives[class_id]
            fp = self.cumulative_false_positives[class_id]

            cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0) # 1D array with shape `(num_predictions,)`
            cumulative_recall = tp / self.num_gt_per_class[class_id] # 1D array with shape `(num_predictions,)`

            cumulative_precisions.append(cumulative_precision)
            cumulative_recalls.append(cumulative_recall)

        self.cumulative_precisions = cumulative_precisions
        self.cumulative_recalls = cumulative_recalls

    def compute_average_precisions(self, mode='sample', num_recall_points=11):
        '''
        Computes the average precision for each class.

        Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling)
        and post-2010 (integration) algorithm versions.

        Note that `compute_precision_recall()` must be called before calling this method.

        Arguments:
            mode (str, optional): Can be either 'sample' or 'integrate'. In the case of 'sample', the average precision will be computed
                according to the Pascal VOC formula that was used up until VOC 2009, where the precision will be sampled for `num_recall_points`
                recall values. In the case of 'integrate', the average precision will be computed according to the Pascal VOC formula that
                was used from VOC 2010 onward, where the average precision will be computed by numerically integrating over the whole
                preciscion-recall curve instead of sampling individual points from it. 'integrate' mode is basically just the limit case
                of 'sample' mode as the number of sample points increases. For details, see the references below.
            num_recall_points (int, optional): Only relevant if mode is 'sample'. The number of points to sample from the precision-recall-curve
                to compute the average precisions. In other words, this is the number of equidistant recall values for which the resulting
                precision will be computed. 11 points is the value used in the official Pascal VOC pre-2010 detection evaluation algorithm.
            verbose (bool, optional): If `True`, will print out the progress during runtime.
            ret (bool, optional): If `True`, returns the average precisions.

        Returns:
            None by default. Optionally, a list containing average precision for each class.

        References:
            http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
        '''

        if (self.cumulative_precisions is None) or (self.cumulative_recalls is None):
            raise ValueError("Precisions and recalls not available. You must run `compute_precision_recall()` before you call this method.")

        if not (mode in {'sample', 'integrate'}):
            raise ValueError("`mode` can be either 'sample' or 'integrate', but received '{}'".format(mode))

        average_precisions = [0.0]

        # Iterate over all classes.
        for class_id in range(1, self.n_classes + 1):

            print("Computing average precision, class {}/{}".format(class_id, self.n_classes))

            cumulative_precision = self.cumulative_precisions[class_id]
            cumulative_recall = self.cumulative_recalls[class_id]
            average_precision = 0.0

            if mode == 'sample':

                for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):

                    cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]

                    if cum_prec_recall_greater_t.size == 0:
                        precision = 0.0
                    else:
                        precision = np.amax(cum_prec_recall_greater_t)

                    average_precision += precision

                average_precision /= num_recall_points

            elif mode == 'integrate':

                # We will compute the precision at all unique recall values.
                unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall, return_index=True, return_counts=True)

                # Store the maximal precision for each recall value and the absolute difference
                # between any two unique recal values in the lists below. The products of these
                # two nummbers constitute the rectangular areas whose sum will be our numerical
                # integral.
                maximal_precisions = np.zeros_like(unique_recalls)
                recall_deltas = np.zeros_like(unique_recalls)

                # Iterate over all unique recall values in reverse order. This saves a lot of computation:
                # For each unique recall value `r`, we want to get the maximal precision value obtained
                # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
                # values after a given iteration, then in the next iteration, in order compute the maximal
                # precisions for the last `l > k` recall values, we only need to compute the maximal precision
                # for `l - k` recall values and then take the maximum between that and the previously computed
                # maximum instead of computing the maximum over all `l` values.
                # We skip the very last recall value, since the precision after between the last recall value
                # recall 1.0 is defined to be zero.
                for i in range(len(unique_recalls)-2, -1, -1):
                    begin = unique_recall_indices[i]
                    end   = unique_recall_indices[i + 1]
                    # When computing the maximal precisions, use the maximum of the previous iteration to
                    # avoid unnecessary repeated computation over the same precision values.
                    # The maximal precisions are the heights of the rectangle areas of our integral under
                    # the precision-recall curve.
                    maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]), maximal_precisions[i + 1])
                    # The differences between two adjacent recall values are the widths of our rectangle areas.
                    recall_deltas[i] = unique_recalls[i + 1] - unique_recalls[i]

                average_precision = np.sum(maximal_precisions * recall_deltas)

            average_precisions.append(average_precision)

        self.average_precisions = average_precisions

    def compute_mean_average_precision(self):
        '''Computes the mean average precision over all classes, except the background class.'''

        self.mean_average_precision = np.average(self.average_precisions[1:])