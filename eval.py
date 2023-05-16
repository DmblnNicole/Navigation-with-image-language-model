import cv2
import evaluate
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import timeit
from PIL import Image
from tqdm import tqdm
from models import ClipSegSD, ClipSegSAM
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import show_mask, combine, image2bitmap, get_GTmask_from_filename, \
    compute_metric, save_metric_for_one_pair, save_metric_for_one_pair_with_SD_output
from evaluation import *

def main(mode : str):
    if mode == 'sd':
        pipeline =  ClipSegSD(
            data_path='./hike/edge', 
            word_mask='A bright photo of a path through the forest',
            sd_prompt='A bright picture of a narrow footpath',
            obstacle_prompt='A dull photo of bulky or voluminous obstacles that are bigger than 50 centimeters'
        )

        files = pipeline.loadData()
        images_dir = './processed_hike/testing_new_pipeline_structure'

        #compute metric for single mask/ground_truth pairs
        for file in tqdm(files):
            mask_path, mask_obstacle, mask_combined, mask_refined, stable_diffusion_output, init_image = pipeline(file)
            metric_for_one_pair = compute_metric([mask_refined], [file])
            save_metric_for_one_pair_with_SD_output(file, init_image, stable_diffusion_output, mask_refined, metric_for_one_pair, title='mask_refined', images_dir=images_dir)
            # masks.append(mask_refined)
            out = combine(init_image, mask_refined)
            out.save(f'{images_dir}/{file}')

    elif 'sam':
        positive = range(2,10)
        negative = range(3,15)
        pipeline = ClipSegSAM(
            data_path='./youtube100',
            word_mask='A bright photo of a road to walk on',
            positive_samples=3,
            negative_samples=10
        )
        files = pipeline.loadData()
        images_dir = './metrics/youtube100'
        sam_masks = []

        for file in tqdm(files):
            init_image, mask_path, sam_mask, coords, labels = pipeline(file)
            sam_masks.append(sam_mask)
            combined_image = combine(init_image, sam_mask, coords=coords, labels=labels)
            metric_for_one_pair = compute_metric([sam_mask], [file])
            save_metric_for_one_pair_sam(
                filename = file, 
                metric = metric_for_one_pair, 
                combined_image = combined_image, 
                images_dir = images_dir, 
                prompt = 'A bright photo of a road to walk on')
            #out.save(f'{images_dir}/{file}')

        metric = compute_metric(sam_masks, files)
        print("metric: ", metric)

if __name__ == '__main__':
    main('sam')

# for i, mask in enumerate(masks):
    

# compute overall metric for all images in hike    
# metric = compute_metric(masks, files)
# # save to text file
# with open('mean_iou.txt', 'a') as f:  # Open the file in append mode
#     f.write(f"Metrics for mask_refined (SD with mask_path and seed) and ground truth pairs of hike dataset: {metric}\n")

# print(metric)



