import random
import time
import os
import numpy as np
import cv2
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock
from collections import Counter

from .bbox_collider import findAndCollide
from .evaluate_map import *


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    torch.cuda.empty_cache()
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            torch.cuda.empty_cache()
            start_compute_time = time.time()
            # print("input type {}".format(type(inputs)))
            print("inputs size {}, file_name -   {}".format(len(inputs), inputs[0]['file_name']))
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time

            # image_id = input["image_id"]
            # instances = output["instances"].to(self._cpu_device)
            # boxes = instances.pred_boxes.tensor.numpy()
            # scores = instances.scores.tolist()
            # classes = instances.pred_classes.tolist()
            # for box, score, cls in zip(boxes, scores, classes):

            img_path = inputs[0]['file_name']
            boxes = outputs[0]['instances'].pred_boxes.tensor
            scores = outputs[0]['instances'].scores.unsqueeze(0).T
            labels = outputs[0]['instances'].pred_classes.unsqueeze(0).T
            
            # print("Input    {}".format(inputs[0]['file_name']))
            # print("outputs    {}".format(outputs[0]['instances']))
#             print(boxes.shape)
#             print(labels.shape)
#             print(scores.shape)
    
            outputs_list = torch.hstack([boxes, labels, scores]).tolist()
            threshold_col = 0.5
#             outputs_reduced = np.asarray(findAndCollide(rectangles=outputs_list, threshold=threshold_col, box_format='voc'))
#             boxes_reduced = torch.Tensor(outputs_reduced[:, :4])
#             scores_reduced = torch.Tensor(outputs_reduced[:, 5])
#             labels_reduced = torch.Tensor(outputs_reduced[:, 4])
            
#             outputs[0]['instances'].pred_boxes.tensor = boxes_reduced
#             outputs[0]['instances'].scores = scores_reduced
#             outputs[0]['instances'].pred_classes = labels_reduced
            evaluator.process(inputs, outputs)
            
            img = cv2.imread(img_path)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            # fontColor = (0, 255, 0)
            thickness = 4
            lineType = 2
            threshold = 0.5    
            
#             print(len(boxes)==len(boxes))

#             for num, box in enumerate(boxes):
                
#                 if scores[num]>=threshold:
#                     # print("box  {}".format(box))
#                     pt1 = (int(box[0].item()), int(box[1].item()))
#                     pt2 = (int(box[2].item()), int(box[3].item()))
#                     # print("pt1, pt2  {},  {}".format(pt1, pt2))
#                     bottomLeftCornerOfText = (pt1[0], pt1[1]+20)
#                     str1 = str(int(labels[num].item())) + "  " + str(round(scores[num].item(), 2))

#                     random.seed()
#                     color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#                     cv2.putText(img, str1,
#                                 bottomLeftCornerOfText,
#                                 font,
#                                 fontScale,
#                                 color1,
#                                 thickness,
#                                 lineType)
#                     cv2.rectangle(img, pt1, pt2, color=color1)
                    
#             basedir = f"datasets/results_{threshold}/{threshold_col}/"
#             if not os.path.exists(basedir):
#                 os.makedirs(basedir, exist_ok=True)
#             fn = os.path.join(basedir, os.path.split(inputs[0]['file_name'])[1])           
# #             cv2.namedWindow("img", 0)
#             cv2.imwrite(fn, img)
# #             cv2.imwrite()
# #             cv2.waitKey(0)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
