import torch
import sys
import os
sys.path.append('./SAM/')
from SAM import SamPredictor, sam_model_registry



class SAMModel: 
    def __init__(self):
        weight_file = "./SAM/MODEL/sam_vit_h_4b8939.pth"
        self.sam = sam_model_registry["vit_h"](checkpoint= weight_file)
        self.model = SamPredictor(self.sam)

    def predict(self, img, text):
        with torch.no_grad():
            return self.model.run_on_web_image(img, text, 0.5)
    
    def set_img(self, img):
        self.model.set_image(img)
