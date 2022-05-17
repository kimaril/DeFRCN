import numpy as np
from torchvision.ops import box_iou
import torch

def coco2voc(box):
    return [box[0], box[1], box[0]+box[2], box[1]+box[3]]

def area(box):
    return (box[2]-box[0])*(box[3]-box[1])
    
def intersection_over_minimal(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    intersection = [x_left, y_top, x_right, y_bottom]

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    iom = area(intersection) / min(area(bb1), area(bb2))
    assert iom >= 0.0
    assert iom <= 1.0
    return iom

def findCollisions(rectangles, threshold=0.1, box_format="coco"):
    collisions = []
    for rectangle in rectangles:
        if len(rectangle)==1: # попадаются вложенные боксы [[]], пока непонятно, почему
            rectangle = np.asarray(rectangle[0]).flatten()
        collisions_indexes = []
        index = 0
        if box_format=="coco":
            rectangle_voc = coco2voc(rectangle)[:4]
        else:
            rectangle_voc = rectangle[:4]
        for currentColission in collisions:
            for rect in currentColission:
                print(len(rect))
                if len(rect)==1:
                    rect = np.asarray(rect[0]).flatten()
                if box_format=="coco":
                    rect_voc = coco2voc(rect)[:4]
                else:
                    rect_voc = rect[:4]
                try:    
                    iou = box_iou(torch.as_tensor([rect_voc]), torch.as_tensor([rectangle_voc]))
                except IndexError:
                    print(rect_voc)
                    print(rectangle_voc)
                iom = intersection_over_minimal(rect_voc, rectangle_voc)
                if ((iou>=threshold) or (iom>=threshold)) and rectangle[4]==rect[4]:
                    collisions_indexes.append(index)
                    break
            index+=1

        if len(collisions_indexes) == 0:
    #         this rectangle collides with none and should be appened to collisions array
            collisions.append([rectangle])
        elif len(collisions_indexes) >= 1:
    #         there is just one collision, we can merge the same
            collisions[collisions_indexes[0]].append(rectangle)

    #         now we have got multiple collisions, so we need to merge all the collisions with the first one
    #         and remove the colission ones
            for i in range(1, len(collisions_indexes)):
    #             we use - (i - 1) because we will remove the collision once we merge it
    #             so after each merge the collision index actually shift by -1

                new_index = collisions_indexes[i] - (i - 1)
    #             extend the first colliding array with the new collision match
                collisions[collisions_indexes[0]].append(collisions[new_index])

    #             now we remove the element from our collision since it is merged with some other
#                 print(collisions)
#                 print(new_index)
                collisions.pop(new_index)
    return collisions

def collide(rectangles, box_format="coco"):
    rectangles = np.array([r if len(r)>1 else r[0] for r in rectangles], dtype=object)    
#     print()
#     print(rectangles)
    label_idx = rectangles[:, 5].argmax()
    label = rectangles[label_idx, 4]
    score = rectangles[label_idx, 5]
    if box_format=="voc":
        left = rectangles[:, 0].min()
        top = rectangles[:, 1].min()
        right = rectangles[:, 2].max()
        bottom = rectangles[:, 3].max()
        return [left, top, right, bottom, label, score]
    else:
        left = rectangles[:, 0].min()
        top = rectangles[:, 1].min()
        right = (rectangles[:, 0]+rectangles[:, 2]).max()
        bottom = (rectangles[:, 1]+rectangles[:, 3]).max()
        width = right - left
        height = bottom - top
        return [left, top, width, height, label, score]

def findAndCollide(rectangles, threshold=0.5, box_format="coco"):
    '''
    Args:
        rectangles: list of bounding boxes with label and score for each box in the following format:
                    COCO: [x_top, y_top, width, height, label, score]
                    VOC: [x_top, y_top, x_bottom, y_bottom, label, score]
        threshold: IoU score threshold, bboxes with IoU>=threshold will be merged
        box_format: either ``coco`` or ``voc``, other values will cause AssertionError
    '''
    assert box_format in ["coco", "voc"]
    collisions = findCollisions(rectangles, threshold=threshold, box_format=box_format)
    return [collide(c, box_format=box_format) for c in collisions]