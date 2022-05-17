from typing import List
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm

def coco2pvoc(bbox):
    xtl, ytl = bbox[0], bbox[1]
    xbr = bbox[0] + bbox[2]
    ybr = bbox[1] + bbox[3]
    return [xtl, ytl, xbr, ybr]

class DetectionEvalStat:
    """
    Detection evaluation statistics, used to collect and count indices
    of true and false detection instances.
    """

    def __init__(self):
        """
        Properties of DetectionEvalStat class are lists.
        Each time predicted bbox is checked against reference bbox,
        the image idx of this bbox appends to corresponding list.
        I.e. if bbox does not correspond to any ground truth bboxes on image,
        the index of this image is appended to fp list.
        """
        self.tp = []  # here we store TP detections
        self.fn = []  # here we store FN detections
        self.fp = []  # here we store FP detections
        self.ref = []  # here we store reference indices = all indices of reference aka test dataset
        self.det = []  # here we collect all detections

    def prec(self):
        if len(self.det) > 0:
            return len(self.tp) / len(self.det)
        return 0

    def recall(self):
        if len(self.det) > 0:
            return len(self.tp) / len(self.ref)
        return 0

    def f1(self):
        prec = self.prec()
        rec = self.recall()
        if prec > 0 or rec > 0:
            f1 = 1 * prec * rec / (prec + rec)
            return f1
        return 0

    def stats2dict(self):
        return {
            "tp": len(self.tp),
            "fn": len(self.fn),
            "fp": len(self.fp),
            "ref": len(self.ref),
            "det": len(self.det),
            "prec": self.prec(),
            "rec": self.recall(),
            "f1": self.f1()
        }


    def __repr__(self):
        return f"stat: tp={len(self.tp)}, fp={len(self.fp)}, fn={len(self.fn)}," + \
               f"ref={len(self.ref)}, det={len(self.det)}, prec={round(self.prec(), 3)},\
                recall={round(self.recall(), 3)}"


def get_total_stats(class_res):
    total_tp = sum([len(_.tp) for _ in class_res])
    total_fp = sum([len(_.fp) for _ in class_res])
    total_fn = sum([len(_.fn) for _ in class_res])
    total_ref = sum([len(_.ref) for _ in class_res])
    total_det = sum([len(_.det) for _ in class_res])
    # print(class_res)

    total_precision = total_tp / total_det
    total_recall = total_tp / total_ref
    # accuracy = 1 - (total_fp + total_fn) / total_det
    f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    return total_precision, total_recall, f1,


class Evaluator:
    def __init__(self, preds, test_dataset, num_classes):
        """
        preds: 
        test_dataset: 
        NOTE: scores, labels and boxes must be torch.Tensor's, detached and moved to CPU before evaluation
        """
        self.preds = preds
        self.test_dataset = test_dataset
        self.N = len(self.test_dataset)
        self.num_classes = num_classes

    def eval_preds(self, score_threshold=0.5, iou_threshold=0.5):
        """
        Evaluate predictions wrt provided self.test_dataset
        :param score_threshold: classification confidence score to consider bbox relevant
        :param iou_threshold: IOU threshold to consider predicted bbox matching with GT
        (GT = ground truth)

        :return: statistics objects
        """
        stats = [DetectionEvalStat() for x in range(self.num_classes)]
        #     print(len(stats))

        for idx in range(self.N):
            sample = self.test_dataset[idx][-1]
            pred = self.preds.iloc[idx]
            # print(sample)
            # print(pred)

            scores = torch.tensor(pred['scores'])
            labels = torch.tensor(pred['labels'])
            boxes = pred['boxes']
            # boxes = torch.tensor([coco2pvoc(b) for b in boxes])
            boxes = torch.tensor(boxes)

            score_filter = torch.where(scores > score_threshold)

            labels = labels[score_filter]
            boxes = boxes[score_filter]
            #         print(sample[-1])
            ref_boxes = sample['boxes']
            ref_labels = sample['labels']

            for i in range(labels.shape[0]):
                #             print(stats)
                stats[labels[i]].det.append(idx)

            # print(ref_labels)
            for i in range(ref_labels.shape[0]):
                stats[ref_labels[i]].ref.append(idx)

            if ref_boxes.shape[0] == 0 or boxes.shape[0] == 0:
                continue

            m_iou = torchvision.ops.generalized_box_iou(ref_boxes, boxes)

            matched_test = set()

            for ref_idx in range(ref_boxes.shape[0]):
                ref_overlap = False

                for test_idx in range(boxes.shape[0]):

                    if m_iou[ref_idx][test_idx] > iou_threshold:
                        ref_overlap = True  # if ref has been overlapped by any other class, then match
                        matched_test.add(test_idx)

                        if ref_labels[ref_idx] == labels[test_idx]:
                            stats[ref_labels[ref_idx]].tp.append(idx)

                # ref doesn't overlap with any test
                if not ref_overlap:
                    stats[ref_labels[ref_idx]].fn.append(idx)

            for test_idx in range(boxes.shape[0]):
                if test_idx not in matched_test:
                    stats[labels[test_idx]].fp.append(idx)

        return stats[1:]

    def eval_detection_mAP(self, iou_threshold=0.5, score_range=None):
        """
        Evaluate model wrt provided test_dataset
        on provided score thresholds range (default is 11-point range)
        :param iou_threshold: intersection over union threshold
        :param score_range: range of detection thresholds

        :return: mAP metric
        """
        if score_range is None:
            score_range = np.arange(0.0, 1.1, 0.1)

        stats: List[dict] = [{x: DetectionEvalStat() for x in score_range}
                             for _ in range(self.num_classes)]
        # num_classes = len(set(np.concatenate(self.test_dataset['labels'].tolist())))
        # stats = [{x: DetectionEvalStat() for x in score_range} for _ in range(num_classes+1)]

        for idx in tqdm(range(self.N)):
            sample = self.test_dataset[idx][-1]
            # sample = self.test_dataset.iloc[idx]

            scores = self.preds.iloc[idx]["scores"]
            labels = self.preds.iloc[idx]["labels"]
            boxes = self.preds.iloc[idx]["boxes"]
            scores = torch.tensor(scores)
            labels = torch.tensor(labels)
            boxes = torch.tensor(boxes)

            ref_boxes = sample['boxes']
            ref_labels = sample['labels']

            for i in range(labels.shape[0]):
                for t in score_range:
                    if scores[i] >= t:
                        stats[labels[i]][t].det.append(idx)

            for i in range(ref_labels.shape[0]):
                for t in score_range:
                    stats[ref_labels[i]][t].ref.append(idx)

            if ref_boxes.shape[0] == 0 or boxes.shape[0] == 0:
                continue

            m_iou = torchvision.ops.generalized_box_iou(ref_boxes, boxes)
            matched_test = set()

            for ref_idx in range(ref_boxes.shape[0]):
                for test_idx in range(boxes.shape[0]):
                    if m_iou[ref_idx][test_idx] > iou_threshold:
                        matched_test.add(test_idx)
                        for t in score_range:
                            if scores[test_idx] >= t:
                                if ref_labels[ref_idx] == labels[test_idx]:
                                    stats[ref_labels[ref_idx]][t].tp.append(idx)
                                else:
                                    stats[labels[test_idx]][t].fn.append(idx)

            for test_idx in range(boxes.shape[0]):
                for t in score_range:
                    if test_idx not in matched_test and scores[test_idx] >= t:
                        stats[labels[test_idx]][t].fp.append(idx)

            ap_stats = []
            for s in stats:
                AP = []
                for i in range(1, len(score_range)):
                    # print(s[score_range[i]].prec(), ', ', (s[score_range[i]].recall() - s[score_range[i-1]].recall()))
                    pr = s[score_range[i]].prec() * (s[score_range[i]].recall() - s[score_range[i-1]].recall())
                    AP.append(pr)
                    # print(AP)
                ap_stats.append(sum(AP))

            mAP = np.mean(ap_stats)

        return mAP, ap_stats