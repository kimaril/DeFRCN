## DeFRCN detector + Descriptor


## Quick Start

**1. Requirements**
* Linux with Python >= 3.6
* CUDA 11.1
* GCC >= 4.9

**2. Installation**
* Clone Code
  ```angular2html
  git clone https://gitlab.iqmen.ru/kim/defrcn.git
  cd DeFRCN
  ```
* Virtual environment
  ```angular2html
  virtualenv defrcn
  cd defrcn
  source ./bin/activate
  ```
* PyTorch 1.6.0 with CUDA 10.1 
  ```shell
  pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  ```
* Detectron2
  ```angular2html
  python3 -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
  ```
* Install other requirements. 
  ```angular2html
  python3 -m pip install -r requirements.txt
  ```

**3. Run Detection**
File paths.txt should contain list of full paths to images for detection.
  ```angular2html
   python3 inference.py image_list_txt_path query_image_path path_to_ISC21Descriptor_weights --config-file path_to_config

  ```

Results of detection and object embeddings will be saved to results/detection_date-time/result.embeddings.cosine.json file.
