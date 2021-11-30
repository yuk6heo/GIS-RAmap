
from __future__ import absolute_import, division

from davisinteractive import logging
from davisinteractive.metrics import batched_jaccard
from davisinteractive.utils.operations import bezier_curve
from davisinteractive.robot.interactive_robot import InteractiveScribblesRobot
from davisinteractive.evaluation.service import EvaluationService

import time
import numpy as np



ROBOT_DEFAULT_PARAMETERS = {
    'kernel_size': .2,
    'max_kernel_radius': 16,
    'min_nb_nodes': 4,
    'nb_points': 1000
}


class Interactrobot(InteractiveScribblesRobot):
    def __init__(self,
                 kernel_size=.2,
                 max_kernel_radius=16,
                 min_nb_nodes=4,
                 nb_points=1000):
        """ Robot constructor
        """
        super(Interactrobot, self).__init__()
        if kernel_size >= 1. or kernel_size < 0:
            raise ValueError('kernel_size must be a value between [0, 1).')

        self.kernel_size = kernel_size
        self.max_kernel_radius = max_kernel_radius
        self.min_nb_nodes = min_nb_nodes
        self.nb_points = nb_points

    def interact_singleimg(self,
                 pred_mask,
                 gt_mask,
                 nb_objects=None,):
        """ Interaction of the Scribble robot given a prediction.
        Given the sequence and a mask prediction, the robot will return a
        scribble in the region that fails the most.

        # Arguments
            sequence: String. Name of the sequence to interact with.
            pred_masks: Numpy Array. Array with the prediction masks. It must
                be an integer array with shape (H x W), one-hot vector for multi object
            gt_masks: Numpy Array. Array with the ground truth of the sequence.
                It must have the same data type and shape as `pred_masks`, one-hot vector for multi object
            nb_objects: Integer. Number of objects in the ground truth mask. If
                `None` the value will be infered from `y_true`. Setting this
                value will speed up the computation.
            frame: Integer. Frame to generate the scribble. If not given, the
                worst frame given by the jaccard will be used.

        # Returns
            dict: Return a scribble (default representation).
        """
        robot_start = time.time()

        predictions = np.asarray(pred_mask, dtype=np.int)
        annotations = np.asarray(gt_mask, dtype=np.int)

        if nb_objects is None:
            obj_ids = np.unique(annotations)
            obj_ids = obj_ids[(obj_ids > 0) & (obj_ids < 255)]
            nb_objects = len(obj_ids)

        obj_ids = [i for i in range(nb_objects + 1)]
        # Infer height and width of the sequence
        h, w = annotations.shape
        img_shape = np.asarray([w, h], dtype=np.float)

        pred, gt = predictions, annotations

        scribbles = []

        for obj_id in obj_ids:
            logging.verbose(
                'Creating scribbles from error mask at object_id={}'.format(
                    obj_id), 2)
            start_time = time.time()
            error_mask = (gt == obj_id) & (pred != obj_id)
            if error_mask.sum() == 0:
                logging.info(
                    'Error mask of object ID {} is empty. Skip object ID.'.
                        format(obj_id))
                continue

            # Generate scribbles
            skel_mask = self._generate_scribble_mask(error_mask)
            skel_time = time.time() - start_time
            logging.verbose(
                'Time to compute the skeleton mask: {:.3f} ms'.format(
                    skel_time * 1000), 2)
            if skel_mask.sum() == 0:
                continue

            G, P = self._mask2graph(skel_mask)
            mask2graph_time = time.time() - start_time - skel_time
            logging.verbose(
                'Time to transform the skeleton mask into a graph: ' +
                '{:.3f} ms'.format(mask2graph_time * 1000), 2)

            t_start = time.time()
            S = self._acyclics_subgraphs(G)
            t = (time.time() - t_start) * 1000
            logging.verbose(
                'Time to split into connected components subgraphs ' +
                'and remove the cycles: {:.3f} ms'.format(t), 2)

            t_start = time.time()
            longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
            longest_paths = [P[idx] for idx in longest_paths_idx]
            t = (time.time() - t_start) * 1000
            logging.verbose(
                'Time to compute the longest path on the trees: {:.3f} ms'.
                    format(t), 2)

            t_start = time.time()
            scribbles_paths = [
                bezier_curve(p, self.nb_points) for p in longest_paths
            ]
            t = (time.time() - t_start) * 1000
            logging.verbose(
                'Time to compute the bezier curves: {:.3f} ms'.format(t), 2)

            end_time = time.time()
            logging.verbose(
                'Generating the scribble for object id {} '.format(obj_id) +
                'took {:.3f} ms'.format((end_time - start_time) * 1000), 2)
            # Generate scribbles data file
            for p in scribbles_paths:
                p /= img_shape
                path_data = {
                    'path': p.tolist(),
                    'object_id': int(obj_id),
                    'start_time': start_time,
                    'end_time': end_time
                }
                scribbles.append(path_data)

        scribbles_data = {'scribbles': scribbles,}

        t = time.time() - robot_start
        logging.info(('The robot took {:.3f} s to generate all the '
                      'scribbles for {} objects.').format(
            t, nb_objects))


        return scribbles_data
