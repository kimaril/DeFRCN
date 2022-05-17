from PIL import Image
import numpy as np
# hello
import torch
import json
import os
import streamlit as st

from torchvision.utils import draw_bounding_boxes
from torchvision import transforms as T
from inference import *

st.title("DeFRCN Model Demo")
st.text("DON\'T TOUCH RERUN BUTTON")

to_pil = T.ToPILImage()
to_tensor = T.ToTensor()

config_file = "configs/voc/defrcn_main_1shot_seed0.yaml"

opt_configs = ["MODEL.WEIGHTS", 
"checkpoints/voc/1/defrcn_fsod_r101_novel2/fsrw-like/10shot_seed0_repeat2/model_0007799.pth",
"TEST.PCB_MODELPATH", 
"data/ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth"]

predictor = configure_model(config_file, opt_configs)

with open("categories/categories_data.4_classes.json", mode="r", encoding="utf-8") as f:
    categories_data = json.loads(f.read())
    
threshold = st.sidebar.slider(
    label='Confidence Score Threshold', 
    min_value=0.0, 
    max_value=1.0,
    value=0.3,
    step=0.1,)
    
input_image = st.file_uploader("Choose input image...", type=['png','jpg', 'jpeg', 'bmp'])

if input_image is not None:
#     image = cv2.imread(input_image)
    image = np.array(Image.open(input_image).convert("RGB"))
    
    clicked = st.button('Predict')
    if clicked:
        prediction = predictor(image)
        prediction = decode_defrcn(prediction, collision_threshold=0.5, box_format="voc")
        boxes, labels, colors = defrcn2torchvision(prediction, threshold, categories_data)
        imtensor = (to_tensor(image)*255).to(torch.uint8)

        image_w_boxes = to_pil(
            draw_bounding_boxes(
                image=imtensor, 
                boxes=boxes, 
                labels=labels, 
                colors=colors, 
                fill=False, 
                width=3,
                font="/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
                font_size=20
            )
        )
        
        if not os.path.exists("./tmp/"):
            os.makedirs("./tmp/")
        image_w_boxes.save("./tmp/streamlit_result.png")
        
        st.image("./tmp/streamlit_result.png", caption=f'Processed Image', use_column_width=True)

#test
