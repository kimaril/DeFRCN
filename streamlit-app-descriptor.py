from PIL import Image
import numpy as np
import torch
import json
import os
import cv2
import streamlit as st
from inference import DetDescriptor
from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
from inference import *

st.title("Few-shot DeFRCN+Descriptor Model Demo")
st.text("DON\'T TOUCH RERUN BUTTON")

to_pil = T.ToPILImage()
to_tensor = T.ToTensor()

with open("categories/categories_data.4_classes.json", mode="r", encoding="utf-8") as f:
    categories_data = json.loads(f.read())
    
threshold = st.sidebar.slider(
    label='Confidence Score Threshold', 
    min_value=0.0, 
    max_value=1.0,
    value=0.3,
    step=0.1,)


ddesc = DetDescriptor("configs/voc/defrcn_main_1shot_seed0.yaml", 
"/home/kim/juche/projects/ISC21-Descriptor-Track-1st/exp/v107/train/checkpoint_0009.pth.tar", 
threshold, 0.7)
    
input_image = st.file_uploader("Choose input image...", type=['png','jpg', 'jpeg', 'bmp'])

if input_image is not None:
#     image = cv2.imread(input_image)
    image = Image.open(input_image).convert("RGB")
    
    clicked = st.button('Predict')

    topn = st.sidebar.slider(
                        label='Top N', 
                        min_value=3, 
                        max_value=100,
                        value=100,
                        step=2,)

    if clicked:
        top = ddesc.get_top_for_image(json_path="/home/kim/juche/projects/DeFRCN/results/detection_20220517-0020/result.embeddings.logos.json",
                                image_fn=input_image, bbox=None, topn=topn)
        # top = ddesc.get_top_for_image(json_path="./results/detection_20220518-1245/result.embeddings.json",
        #                 image_fn=input_image, bbox=None, topn=topn)

        st.image(image, caption="Original query image", use_column_width=True)

        for i, _ in top.iterrows():
            similarity = round(_.abssim, 2)
            label = int(_.label)
            detection_score = round(_.score, 2)
            str_label = categories_data[label]["label"]
            str_label_score = f"{str_label}, confidence: {detection_score}"
            fn = _.im_fn
            caption = f"Label: {str_label}, Det. Score: {detection_score}, Similarity: {similarity}\nIndex:{i}\nFilename:{fn}"
            
            imtensor = (to_tensor(Image.open(_.im_fn))*255).to(torch.uint8)
            # imtensor = torch.from_numpy((np.asarray(cv2.imread(_.im_fn))*255).astype(np.uint8))

            try:
                color = categories_data[label]["color"]
                box = torch.tensor([_.box])
                imres = draw_bounding_boxes(
                    image=imtensor, 
                    boxes=box, 
                    labels=[str_label_score], 
                    colors=[(255,0,0)], 
                    fill=False, 
                    width=3,
                    font="/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
                    font_size=20
                )
                imres = to_pil(imres)
                # st.image(Image.open(_.im_fn).crop(_.box), caption=caption, use_column_width=True)
                st.image(imres, caption=caption, use_column_width=True)

            except TypeError as e:
                print(e)
                print(fn)
                error_caption = "PIL.ImageDraw failed, showing full image without bounding boxes\n"
                st.image(Image.open(_.im_fn), caption=error_caption+caption, use_column_width=True)
                continue