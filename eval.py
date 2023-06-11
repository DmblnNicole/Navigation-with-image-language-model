from matplotlib import pyplot as plt
import numpy as np
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
            metric_for_one_pair, _ = compute_metric([mask_refined], [file])
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
            positive_samples=9,
            negative_samples=11
        )
        files = pipeline.loadData()
        """ computes and saves metrics for whole dataset """
        images_dir = './test'
        sam_masks = []
        for file in tqdm(files):

            #prompt once
            multimask_output = True
            mask_input = None
            init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)

            # prompt twice
            multimask_output = False
            mask_input = logits[np.argmax(scores), :, :]
            init_image, mask_path, sam_mask, _, logits, coords, labels = pipeline(file, multimask_output, mask_input)
            
            # save image to folder
            combined_image = combine(init_image, sam_mask, coords=coords, labels=labels)
            combined_image.save(f'{images_dir}/{file}')
            sam_masks.append(sam_mask)

        # compute metric for dataset
        metric, _ = compute_metric(sam_masks, files)
        print("metric: ", metric)
        # save metric to text file
        with open('mean_iou.txt', 'a') as f:
            f.write(f"{images_dir}: {metric}\n")
        

if __name__ == '__main__':
    main('sam')





