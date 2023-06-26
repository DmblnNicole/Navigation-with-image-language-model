import numpy as np
from tqdm import tqdm
from models import ClipSegSD, ClipSegSAM
from utils import *
import os

def main(mode : str, visualize: bool):
    if mode == 'sd':
        pipeline =  ClipSegSD(
            data_path='../data/images/hike/edge', 
            word_mask='A bright photo of a path through the forest',
            sd_prompt='A bright picture of a narrow footpath',
            obstacle_prompt='A dull photo of bulky or voluminous obstacles that are bigger than 50 centimeters'
        )

        files = pipeline.loadData()
        if visualize:
            images_dir = '../output'
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            else:
                print("The folder already exists.")
        GT_dir = '../data/GT/GT_hike'
        masks = []

        #compute metric for single mask/ground_truth pairs
        for file in tqdm(files):
            _, _, _, final_mask, stable_diffusion_output, init_image= pipeline(file)
            masks.append(final_mask)
            if visualize:
                # visualize final mask that is obtained by prompting ClipSeg with stable_diffusion_output
                out = combine(init_image, final_mask)
                out.save(f'{images_dir}/{file}')
        # compute metric for dataset
        metric, _ = compute_metric(GT_dir, masks, files)
        print("metric: ", metric)
        with open('metrics.txt', 'a') as f:
            f.write(f"Stable Diffusion: {metric}\n")

    elif 'sam':
        positive = range(2,10)
        negative = range(3,15)
        pipeline = ClipSegSAM(
            data_path='../data/images/hike/edge',
            word_mask='A bright photo of a road to walk on',
            positive_samples=9,
            negative_samples=11
        )
        """ computes and saves metrics for whole dataset """
        files = pipeline.loadData()
        if visualize:
            images_dir = '../output'
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            else:
                print("The folder already exists.")
        GT_dir = '../data/GT/GT_hike'
        sam_masks = []
        for file in tqdm(files):

            #prompt once
            multimask_output = True
            mask_input = None
            init_image, _ , sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)

            # prompt twice
            multimask_output = False
            mask_input = logits[np.argmax(scores), :, :]
            init_image, _ , sam_mask, _, logits, coords, labels = pipeline(file, multimask_output, mask_input)
            
            sam_masks.append(sam_mask)
            # save image to folder
            if visualize:
                combined_image = combine(init_image, sam_mask, coords=coords, labels=labels)
                combined_image.save(f'{images_dir}/{file}')
                

        # compute metric for dataset
        metric, _ = compute_metric(GT_dir, sam_masks, files)
        print("metric: ", metric)
        # save metric to text file
        with open('metrics.txt', 'a') as f:
            f.write(f"SAM: {metric}\n")
        

if __name__ == '__main__':
    main('sd', visualize=True)





