import os
import re
import sys
import uuid
from io import BytesIO
from pathlib import Path

import clip
import cv2
import numpy as np
import PIL
import requests
import torch
import transformers
from diffusers import DiffusionPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torch import autocast
from torchvision import transforms
from segment_anything import sam_model_registry, SamPredictor
from clipseg.models.clipseg import CLIPDensePredT
from utils import bitmap2image

class ClipSegBase():

    """
    Base Class template for adding more models into clipseg pipeline
    """

    def __init__(self, data_path : str, word_mask : str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_path = data_path
        self.word_mask = word_mask
        self.initClipseg()
    
    def initClipseg(self):
        """ Initialize pretrained clipseg model """
        self.clipseg_model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
        self.clipseg_model.eval()
        self.clipseg_model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)
    
    def preprocess(self):
        """ Preprocess image for ClipSeg"""
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Resize((512, 512)),
                    ])
        return transform
    
    def promptClipSeg(self, image, word_mask):
        """ Extract binary mask from 'image' based off 'word_mask' """
        transform = self.preprocess()
        preprocessed_image = transform(image).unsqueeze(0)
        word_masks = [word_mask]
        with torch.no_grad():
            preds = self.clipseg_model(preprocessed_image.repeat(len(word_masks),1,1,1), word_masks)[0]
        init_image = image.convert('RGB').resize((512, 512))
        filename = f"{uuid.uuid4()}.png"
        plt.imsave(filename,torch.sigmoid(preds[0][0]))
        img2 = cv2.imread(filename)
        gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
        mask = Image.fromarray(np.uint8(bw_image)).convert('RGB')
        os.remove(filename)
        return mask, init_image
    
    def preprocess(self):
        """ Preprocess image for ClipSeg"""
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Resize((512, 512)),
                    ])
        return transform

    def loadData(self):
        """ Load image data to pass through ClipSeg"""

        files = os.listdir(self.data_path)
        re_pattern = re.compile('.+?(\d+)\.([a-zA-Z0-9+])')
        try:
            files = sorted(files, key=lambda x: str(re_pattern.match(x).group()))
        except AttributeError:
            files = sorted(files, key=lambda x: str(re_pattern.match(x)))
        return files
    
    def saveImage(self, image, filename, images_dir):
        cv2.imwrite(os.path.join(images_dir, filename), np.asarray(image))
    

class ClipSegSD(ClipSegBase):
    def __init__(self, data_path: str, word_mask: str, sd_prompt : str, obstacle_prompt : str) -> None:
        super().__init__(data_path, word_mask)

        self.initStableDiffusionInpainting()
        self.word_mask = word_mask
        self.obstacle_prompt = obstacle_prompt
        self.sd_prompt = sd_prompt
    
    
    def initStableDiffusionInpainting(self):
        """ Initialize pretrained stable diffusion inpainting model """
        self.diffusion_model = DiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    revision="fp16",
                    torch_dtype=torch.float16,).to(self.device)

    def generateLatents(self):
        """ Generate latent codes with random sequence fixed with seed to reproduce stable diffusion outputs.
            Adapted from https://github.com/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb """
        device = "cuda"
        generator = torch.Generator(device=device)
        seed = 42
        # same sequence of random numbers is generated every time the code is run
        generator.manual_seed(seed)
        latents = torch.randn(
            # values in the tensor are drawn from a normal distribution with mean 0 and standard deviation 1
            # using the torch.Generator object with fixed seed 42
            (1, 4, 512 // 8, 512 // 8),
            generator = generator,
            device = device)
        return latents
        

    def promptStableDiffusionInpainting(self, prompt, image, clipseg_mask):
        """ Fill masked out region of 'image' with content based on 'prompt' """
        if prompt != None:
            with autocast("cuda"):
                output = self.diffusion_model(prompt=prompt, image=image, mask_image=clipseg_mask,latents=self.generateLatents())
        else:
            output = None
        return output
    
    def combineObstacleAndPathMasks(self, mask_obstacle, mask_path):
        """ Adds 'mask_obstacle' and 'mask_path' to get a combined masked of traversable area """

        mask_obstacle = np.asarray(mask_obstacle)
        mask_path = np.asarray(mask_path)
        mask = np.copy(mask_path)
        white_path_pixels = (mask_path == [255,255,255]).all(axis=2)
        white_obstacle_pixels = (mask_obstacle == [255,255,255]).all(axis=2)
        mask[white_path_pixels & white_obstacle_pixels] = [0,0,0]
        combined_mask = Image.fromarray(np.uint8(mask)).convert('RGB')

        return combined_mask
           
    
    def forwardPass(self, file):
        """ Generates 4 masks with clipseg and one inpainted image with stable diffusion inpainting.
            - prompt clipseg with image and mask_path
            - prompt clipseg with image and mask_obstacle
            - combine both masks
            - prompt stable diffusion with image, combined mask and prompt
            - prompt clipseg with stable diffusion output and prompt """
        
        image = Image.open(os.path.join(self.data_path,file))
        mask_path, init_image = self.promptClipSeg(image=image, word_mask=self.word_mask)
        mask_obstacle, _ = self.promptClipSeg(image=image, word_mask=self.obstacle_prompt)
        combined_mask = self.combineObstacleAndPathMasks(mask_path=mask_path, mask_obstacle=mask_obstacle)
        stable_diffusion_output = self.promptStableDiffusionInpainting(prompt=self.sd_prompt, image=init_image, clipseg_mask=combined_mask)
        stable_diffusion_output = stable_diffusion_output.images[0]
        refined_mask_path, _ = self.promptClipSeg(image=stable_diffusion_output, word_mask=self.sd_prompt)    
          
        return mask_path, mask_obstacle, combined_mask, refined_mask_path, stable_diffusion_output, init_image

    def saveMasks(self, mask_path, mask_obstacle, combined_mask, refined_mask_path, images_dir, filename):
        plt.figure(figsize=(12,12))
        plt.subplot(2,2,1)
        plt.title(f'Mask path - {self.word_mask_path_1}')
        plt.imshow(np.asarray(mask_path))
        plt.subplot(2,2,2)
        plt.title(f'Mask obstacle - {self.word_mask_obstacle}')
        plt.imshow(mask_obstacle)
        plt.subplot(2,2,3)
        plt.title('Combined Mask')
        plt.imshow(combined_mask)
        plt.subplot(2,2,4)
        plt.title(f'Refined Mask - {self.prompt}')
        plt.imshow(refined_mask_path)
        plt.savefig(f"{images_dir}/{filename}")
            
    
    def saveImageAndMasks(self, init_image, mask, stable_diffusion_output, refined_mask_path, images_dir, filename):
        masked_image = cv2.add(np.asarray(init_image), np.asarray(refined_mask_path))
        plt.figure(figsize=(12,12))
        plt.subplot(2,2,1)
        plt.title('Initial Image')
        plt.imshow(np.asarray(init_image))
        plt.subplot(2,2,2)
        plt.title(f'Mask')
        plt.imshow(mask)
        plt.subplot(2,2,3)
        plt.title(f'SD Output - {self.prompt}')
        plt.imshow(stable_diffusion_output)
        plt.subplot(2,2,4)
        plt.title(f'Masked Image')
        plt.imshow(masked_image)
        plt.savefig(f"{images_dir}/{filename}")
    

class ClipSegSAM(ClipSegBase):
    def __init__(self, data_path: str, word_mask: str) -> None:
        super().__init__(data_path, word_mask)
        sam = sam_model_registry["vit_h"](checkpoint='checkpoints/sam_vit_h_4b8939.pth').to(self.device)
        self.sam = SamPredictor(sam)

    def sample_points(self):
        pass
        
    def forwardPass(self, file):
        image = Image.open(os.path.join(self.data_path,file))
        mask_path, init_image = self.promptClipSeg(image=image, word_mask=self.word_mask)
        self.sam.set_image(np.asarray(init_image))
        # ind = int(np.random.uniform(low=0, high=len(positive_slice)))
        positive_slice = np.argwhere(np.asarray(mask_path)[:,:,0] != 0)
        positive_slice[:,1] = 512 - positive_slice[:,1]
        point = np.array([np.mean(positive_slice, axis=0)]).astype(np.uint8)
        print(point)
        mask_path, _, _ = self.sam.predict(
            point_coords=point,
            point_labels=[1],
            multimask_output=False
        )
        sam_mask = bitmap2image(mask_path[0])
        return init_image, mask_path, sam_mask
    
    