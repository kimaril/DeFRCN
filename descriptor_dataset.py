import os
import random
import json
import cv2
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from PIL import Image

def opencv2pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

class ISCTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        ref_paths,
        ref_paths2,
        transforms,
        num_negatives,
    ):
        self.paths = paths
        self.ref_paths = ref_paths
        self.ref_paths2 = ref_paths2
        self.transforms = transforms
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        ref_path = self.ref_paths[i]
        image = Image.open(path)
        image = self.transforms(image)
        ref_image = Image.open(ref_path)
        ref_image = self.transforms(ref_image)

        js = [random.choice(range(len(self.ref_paths2))) for _ in range(self.num_negatives)]

        ret = [
            image,
            ref_image,
            *[self.transforms(Image.open(self.ref_paths2[j])) for j in js],
        ]

        return i, ret, [j + 1000000 for j in js]


class ISCTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        transforms,
    ):
        self.paths = paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i])
        image = self.transforms(image)
        return i, image

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path,
        data_dir,
        transforms,
    ):
        self.json_path = json_path
        with open(json_path, "r", encoding="utf-8") as f:
            #im_fn, boxes, labels, scores
            self.table = json.load(f)
        self.transforms = transforms
        self.data_dir = data_dir
        self.data = []
        for im in self.table:
            for i, box in enumerate(im["boxes"]):
                d = dict(
                    id=os.path.splitext(os.path.split(im["im_fn"])[-1])[0]+"_"+str(i).zfill(2), 
                    im_fn=im["im_fn"], 
                    box=box,
                    label=im["labels"][i],
                    score=im["scores"][i],
                    )
                self.data.append(d)
        del self.table
        with open("tmp_data_.json", mode="w", encoding="utf-8") as f:
            json.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = cv2.imread(os.path.join(self.data_dir, self.data[i]["im_fn"]))
        box = self.data[i]["box"]
        h, w, c = image.shape
        x1 = max(round(box[0]), 0)
        y1 = max(round(box[1]), 0)
        x2 = min(round(box[2]), w)
        y2 = min(round(box[3]), h)
        image = opencv2pil(image[y1:y2, x1:x2, :])
        image = self.transforms(image)
        return i, image