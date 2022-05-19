from defrcn.config import get_cfg, set_global_cfg
from defrcn.engine import DefaultPredictor
from defrcn.engine import default_argument_parser
from PIL import Image
import numpy as np
import torch
import json
import os
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
import torch.nn.functional as F
from bbox_collider import *
import json
import time
import timm
from descriptor_net import ISCNet
import torch.nn as nn
from desciptor_engine import extract
from descriptor_dataset import opencv2pil
from tqdm import tqdm 
import pandas as pd


to_pil = T.ToPILImage()
to_tensor = T.ToTensor()
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

with open("categories/categories_data.4_classes.json", mode="r", encoding="utf-8") as f:
    categories_data = json.loads(f.read())

preprocesses = [
    T.Resize((int(512 * 1.4142135623730951), int(512 * 1.4142135623730951))),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
]
preprocess = T.Compose(preprocesses)

def configure_model(config_file, opt_configs):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opt_configs)
    cfg.INPUT.FORMAT = "BGR"
    cfg.freeze()
    set_global_cfg(cfg)
    predictor = DefaultPredictor(cfg)
    return predictor

def decode_defrcn(prediction, collision_threshold=0.5, box_format="voc"):
    boxes = prediction["instances"].get("pred_boxes").tensor.detach().cpu()
    scores = prediction["instances"].get("scores").detach().cpu().tolist()
    labels = prediction["instances"].get("pred_classes").detach().cpu().tolist()
    
    decoded = []
    for b, l, s in zip(boxes, labels, scores):
        _ = []
        _.extend(b)
        _.append(l)
        _.append(s)
        decoded.append(_)
    
    if collision_threshold:
        reduced = findAndCollide(decoded, threshold=collision_threshold, box_format=box_format)
        return reduced
    return decoded

def defrcn2torchvision(prediction, threshold, categories_data):
    
    if "instances" in prediction:
        boxes = prediction["instances"].get("pred_boxes").tensor.detach().cpu()
        scores = prediction["instances"].get("scores").detach().cpu().tolist()
        labels = prediction["instances"].get("pred_classes").detach().cpu().tolist()
    else:
        boxes = torch.as_tensor([p[:4] for p in prediction])
        labels = [p[4] for p in prediction]
        scores = [p[-1] for p in prediction]
    
    str_labels = []
    colors = []
    for l, s in zip(labels, scores):
        if s>=threshold:
            str_labels.append("{}: {}".format(categories_data[int(l)]["label"], round(s, 3)))
            colors.append(tuple(categories_data[int(l)]["color"]))
            
    return boxes[:len(str_labels)], labels[:len(str_labels)], scores[:len(str_labels)], str_labels, colors

class DetDescriptor:
    def __init__(self, det_config_path, desc_path, detection_th, bbox_th):
        self.detector = configure_model(config_file=det_config_path, opt_configs=[])
        self.descriptor_weights = desc_path
        self.load_descriptor()
        self.detection_th = detection_th
        self.collision_th = bbox_th
        self.timestr =  time.strftime("%Y%m%d-%H%M")

    def load_descriptor(self):
        backbone = timm.create_model('tf_efficientnetv2_m_in21ft1k', features_only=True, pretrained=True)
        self.descriptor = ISCNet(backbone, p=3.0, eval_p=1.0)
        self.descriptor = nn.DataParallel(self.descriptor)
        state_dict = torch.load(self.descriptor_weights, map_location='cpu')['state_dict']
        self.descriptor.load_state_dict(state_dict, strict=False)
        self.descriptor.eval().cuda()


    def detect(self, image_fn):
        image = np.array(Image.open(image_fn).convert("RGB"))
        prediction = self.detector(image)
        prediction = decode_defrcn(prediction, collision_threshold=self.collision_th, box_format="voc")
        boxes, labels, scores, str_labels, colors = defrcn2torchvision(prediction, self.detection_th, categories_data)
        imtensor = (to_tensor(image)*255).to(torch.uint8)
        return boxes, labels, scores, str_labels, colors, imtensor


    def describe_image_list(self, files_list_txt, out_dir="results/detection"):
        timestr = time.strftime("%Y%m%d-%H%M")
        if not os.path.exists(f"./{out_dir}_{timestr}/"):
            os.makedirs(f"./{out_dir}_{timestr}/", exist_ok=True)

        with open(files_list_txt) as f:
            files_list = [_.strip() for _ in f.readlines()]

        data = []
        
        for image_fn in tqdm(files_list):
            boxes, labels, scores, str_labels, colors, imtensor = self.detect(image_fn)
            data.append(
                dict(
                    im_fn=str(image_fn), 
                    boxes=boxes.tolist(),
                    labels=labels,
                    scores=scores,
                    )
                    )
        json_path = f"./{out_dir}_{timestr}/result.json"
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f)
        extract(json_path=json_path, data_dir=os.path.abspath("./"))
        print("Saved results to: {}".format(os.path.splitext(json_path)[0] + ".embeddings" + ".json"))
        return os.path.splitext(json_path)[0] + ".embeddings" + ".json"


    def describe_query_image(self, image_fn, bbox=None):
        image = Image.open(image_fn).convert("RGB")
        
        # TODO заменить на annot_fn, если известен формат файла аннотации
        if bbox: # pascal voc format
            image = np.array(image)
            h, w, c = image.shape
            x1 = max(round(bbox[0]), 0)
            y1 = max(round(bbox[1]), 0)
            x2 = min(round(bbox[2]), w)
            y2 = min(round(bbox[3]), h)
            image = opencv2pil(image[y1:y2, x1:x2, :])
        image = torch.stack([preprocess(image)]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_big = image
            image = F.interpolate(image_big, size=512, mode='bilinear', align_corners=False)
            image_small = F.interpolate(image, scale_factor=0.7071067811865475, mode='bilinear', align_corners=False)
            f = (
                self.descriptor(image) + self.descriptor(image_small) + self.descriptor(image_big)
                + self.descriptor(T.functional.hflip(image))
                + self.descriptor(T.functional.hflip(image_small))
                + self.descriptor(T.functional.hflip(image_big))
            )
            f /= torch.linalg.norm(f, dim=1, keepdim=True)
            
        f = f[0].cpu().numpy()
        print(f.shape)
        return f.astype(np.float32)

    def compare_embeddings(self, json_path, image_fn, bbox=None):
        query_image_embedding = self.describe_query_image(image_fn, bbox)
        with open(json_path, "r", encoding="utf-8") as f:
            # data_image_list = json.load(f)
            data_image_list = pd.read_json(f)
        sims = []
        for i, d in data_image_list.iterrows():
            sims.append(cos(torch.as_tensor(d["embedding"]), torch.as_tensor(query_image_embedding)).item())
        data_image_list["sim"] = sims
     
        # with open(os.path.splitext(json_path)[0] + ".cosine" + ".json", mode="w", encoding="utf-8") as f:
            # json.dump(data_image_list, f)
        data_image_list.to_json(os.path.splitext(json_path)[0] + ".cosine" + ".json", force_ascii=False)

        print("Saved cosine similarities to {}".format(os.path.splitext(json_path)[0] + ".cosine" + ".json"))
        return os.path.splitext(json_path)[0] + ".cosine" + ".json"

    def full_cycle(self, files_list_txt, image_fn, bbox=None):
        json_path = self.describe_image_list(files_list_txt)
        res = self.compare_embeddings(json_path, image_fn, bbox=None)
        return res

    def get_top_for_image(self, json_path, image_fn, bbox=None, topn=10):
        cosine_json_path = self.compare_embeddings(json_path=json_path, image_fn=image_fn, bbox=bbox)
        #TODO: убрать pandas, заменить json+sorted
        df = pd.read_json(cosine_json_path)
        print(df.head())
        df["abssim"] = df["sim"].apply(abs)
        top = df.sort_values(by="abssim", ascending=False)[:topn].copy(deep=True)
        return top

def start_inference(config_file, opt_configs, files_list, output_dir="results/detection", threshold=0.5):
    timestr = time.strftime("%Y%m%d-%H%M")
    predictor = configure_model(config_file, opt_configs)
    data = []
    if not os.path.exists(f"./{output_dir}_{timestr}/"):
        os.makedirs(f"./{output_dir}_{timestr}/", exist_ok=True)
    for image_fn in files_list:
        image = np.array(Image.open(image_fn).convert("RGB"))
        prediction = predictor(image)
        prediction = decode_defrcn(prediction, collision_threshold=0.5, box_format="voc")
        boxes, labels, scores, str_labels, colors = defrcn2torchvision(prediction, threshold, categories_data)
        imtensor = (to_tensor(image)*255).to(torch.uint8)

        image_w_boxes = to_pil(
            draw_bounding_boxes(
                image=imtensor, 
                boxes=boxes, 
                labels=str_labels, 
                colors=colors, 
                fill=False, 
                width=3,
                font="/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
                font_size=20
            )
        )
        fn = os.path.split(image_fn)[-1]
        image_w_boxes.save(f"./{output_dir}_{timestr}/{fn}")
        data.append(
            dict(
                im_fn=str(image_fn), 
                boxes=boxes.tolist(),
                labels=labels,
                scores=scores,
            )
        )
        
    with open(f"./{output_dir}_{timestr}/result.json", mode="w", encoding="utf-8") as f:
        json.dump(data, f)
    return os.path.abspath(f"./{output_dir}_{timestr}/result.json")
        
if __name__=='__main__':
    parser = default_argument_parser()

    parser.add_argument('files_listing', metavar='FILES_LIST_TXT', type=str,
                        help='path to txt list of image files for inference')

    parser.add_argument('query_image', metavar='QUERY_IMG', type=str,
                        help='path to query image')
    
    parser.add_argument('desc_path', metavar='DESC_PATH', type=str,
                        help='path to descriptor weights file')

    parser.add_argument('--image-dir', type=str, default='results/detection',
                        help='directory to store images with predictions')

    #TODO: add thresholds to parser
    args = parser.parse_args()
    
    ddescriptor = DetDescriptor(args.config_file, args.desc_path, 0.5, 0.7)
    ddescriptor.compare_embeddings(args.files_listing, args.query_image)
    