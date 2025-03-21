import torch
import os
import sys
import hashlib
import numpy as np
import PyOpenColorIO as OCIO
import OpenImageIO as oiio
import folder_paths
# folder_paths is the module from the comfyui package
from PIL import Image


class LoadOpenEXR:
    @classmethod
    def INPUT_TYPES(s):
        input_path = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.endswith(".exr")]
        return {
            "required":{
                "image": (sorted(files), {"image_upload": True}),
                "crop" : ("BOOLEAN", {"default" : 1, "step" : 1}),
                "crop_height_start": ("INT" , {"default" : 1, "step" : 1}),
                "crop_height_end": ("INT" , {"default" : 512, "step" : 1}),
                "crop_width_start": ("INT" , {"default" : 1, "step" : 1}),
                "crop_width_end" : ("INT" , {"default" : 1, "step" : 1}),
                "resize": ("BOOLEAN", {"default": False}),
                "resize_percentage" : (
                    [
                        "10%",
                        "25%",
                        "50%",
                    ], {
                        "default" : "50%"
                    }
                ),
            },
        },

    RETURN_TYPES  =  ("IMAGE", )
    FUNCTION      =  "read_exr"
    CATEGORY      =  "OpenEXR"

    def read_exr(self, image, crop, crop_height_start, crop_height_end, crop_width_start, crop_width_end, resize, resize_percentage):

        image_path = folder_paths.get_annotated_filepath(image) # return a valid full path to the image

        input = oiio.ImageBuf(image_path)
        pixels = input.get_pixels(oiio.FLOAT) # shape should be (h, w, ch)

        w = pixels.shape[1]
        h = pixels.shape[0]

        if crop and resize:
            raise Exception("Can not crop and resize at the same time!")
        
        elif resize:
            ratio = float(resize_percentage.strip('%'))/100
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            input_mod = oiio.ImageBufAlgo.resize(input, roi = oiio.ROI(0, new_w, 0, new_h, 0, 1, 0, 3))
            pixels = input_mod.get_pixels(oiio.FLOAT)
        
        elif crop:
            input_mod = oiio.ImageBufAlgo.cut(input, roi = oiio.ROI(crop_width_start, crop_width_end, crop_height_start, crop_height_end))
            pixels    = input_mod.get_pixels(oiio.FLOAT)

        result = torch.from_numpy(pixels).float() # change numpy to tensor

        result = torch.unsqueeze(result, 0)

        return (result, )


class SaveOpenEXR:

    @classmethod
    def INPUT_TYPES(s):
        output_path = folder_paths.get_output_directory()
        subfolders  = [os.path.join(output_path, i) for i in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, i))]
