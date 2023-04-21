import cv2
import evaluate
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import timeit
from PIL import Image
from ClipSegStableDiffusionPipeline import ClipSegStableDiffusionPipeline
    

class Evaluate():
    def __init__(self, filenames: list[str], masks: list):
        self.filenames = filenames
        self.masks = masks
    
    def image2bitmap(self, image : Image, dtype=np.bool_) -> np.array:
        image = np.asarray(image, dtype=dtype)
        bitmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool_)
        bitmap = image[:,:,0]
        return bitmap
    
    def compute_metrics(self): 
        gt_masks = []
        resize = transforms.Resize((512,512))
        
        metric = evaluate.load('mean_iou')
        for n in self.filenames:
            a = Image.open('GT_masks/'+n)
            gt_masks.append(self.image2bitmap(resize(a)))

        metrics = metric._compute(
            predictions=self.masks,
            references=gt_masks,
            num_labels=2, 
            ignore_index=255
        )
        
        return metrics

    def measure_time_elapsed(self):
        pass
    
    def show_max_IOU_masks(self):
        pass
    
    def show_min_IOU_masks(self):
        pass
    
    

pipeline = ClipSegStableDiffusionPipeline()
files = pipeline.loadData()
for i, file in enumerate(files):
    mask_path, mask_obstacle, combined_mask, refined_mask_path, stable_diffusion_output = pipeline.forwardPass(file)
    eval = Evaluate(files, [mask_path, mask_obstacle, combined_mask, refined_mask_path])
    metrics = eval.compute_metrics()
    print(metrics)
   
    
    
    


