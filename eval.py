import cv2
import evaluate
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import timeit
from PIL import Image
from models import ClipSegSD
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils import show_mask, combine, image2bitmap, get_GTmask_from_filename, compute_metric, save_metric_for_one_pair, save_metric_for_one_pair_with_SD_output

def main():
    pipeline =  ClipSegSD(
        data_path='./hike/edge', 
        word_mask='A bright photo of a path through the forest',
        sd_prompt='A bright picture of a narrow footpath',
        obstacle_prompt='A dull photo of bulky or voluminous obstacles that are bigger than 50 centimeters'
    )

    files = pipeline.loadData()
    images_dir = './processed_hike/testing_new_pipeline_structure'

    #compute metric for single mask/ground_truth pairs
    for i, file in enumerate(files):
        mask_path, mask_obstacle, mask_combined, mask_refined, stable_diffusion_output, init_image = pipeline.forwardPass(file)
        metric_for_one_pair = compute_metric([mask_refined], [file])
        save_metric_for_one_pair_with_SD_output(file, init_image, stable_diffusion_output, mask_refined, metric_for_one_pair, title='mask_refined', images_dir=images_dir)
        # masks.append(mask_refined)
        out = combine(init_image, mask_refined)
        out.save(f'{images_dir}/{file}')

if __name__ == '__main__':
    main()

# for i, mask in enumerate(masks):
    

# compute overall metric for all images in hike    
# metric = compute_metric(masks, files)
# # save to text file
# with open('mean_iou.txt', 'a') as f:  # Open the file in append mode
#     f.write(f"Metrics for mask_refined (SD with mask_path and seed) and ground truth pairs of hike dataset: {metric}\n")

# print(metric)



