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

    def predict(self, bbox, multimask_output=False):
        masks, scores, _ = self.model.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=multimask_output
        )
        if (multimask_output==False):
            masks = masks.squeeze()
        return masks, scores

    def set_img(self, img):
        self.model.set_image(img)
