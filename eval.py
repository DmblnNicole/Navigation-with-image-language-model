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
from sklearn.metrics import roc_curve, roc_auc_score
import pickle

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
        
        """ - prompt each image 100 times with uniformly sampled points
            - get 100 masks and 100 logits
            - get one heat map by adding all logits and rescale to [0,1]
            - overlay heatmap with image """

        # for file in tqdm(files):
        #     logits_sum = np.zeros((1,256,256))
        #     for i in range(100):
        #         # prompt once
        #         multimask_output = True
        #         mask_input = None
        #         init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
                
        #         # prompt twice
        #         multimask_output = False
        #         mask_input = logits[np.argmax(scores), :, :]
        #         init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
        #         #sam_mask.save(f'./heatmap/multimask/{i}_{files[0]}')
                
        #         # add logits
        #         logits_sum += logits
                
        #     # upsample logits to match input image size
        #     upsampled_logits = cv2.resize(logits_sum[0], (512, 512), interpolation=cv2.INTER_LINEAR)
        #     # normalize logits to [0,1]          Question: or get average logits by logits/100?
        #     normalized_logits = (upsampled_logits - np.min(upsampled_logits)) / (np.max(upsampled_logits) - np.min(upsampled_logits))
        #     # create heatmap
        #     heatmap = cv2.applyColorMap(np.uint8(normalized_logits * 255), cv2.COLORMAP_JET)
        #     overlay = cv2.addWeighted(np.asarray(init_image), 0.5, heatmap, 0.5, 0)
        #     cv2.imwrite(f'{images_dir}/{file}', overlay)
            
        """ get the red and orange mask when prompting the model several times """
        thresholded_logits_masks = []
        num_reprompts = 10
        #images_dir = f'./path_sum_over_logits_{num_reprompts}'
        images_dir = './accumulated_logits_reprompt20'
        sam_masks = []
        for file in tqdm(files):
            # prompt once
            multimask_output = True
            mask_input = None
            init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
            logits_sum = np.zeros((1,256,256))
            for i in range(num_reprompts):
                # prompt twice
                multimask_output = False
                mask_input = logits[np.argmax(scores), :, :]
                init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
                #sam_mask.save(f'./heatmap/multimask/{i}_{files[0]}')
                # add logits
                logits_sum += logits
            logits_mean = logits_sum * (1/num_reprompts)
            # prompt last time with accumulated logits
            mask_input = logits_mean[0]
            init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
            sam_masks.append(sam_mask)
            combined_image = combine(init_image, sam_mask, coords=coords, labels=labels)
            combined_image.save(f'{images_dir}/{file}')

        # 
        # 
        #     # upsample logits to match input image size
        #     upsampled_logits = cv2.resize(logits_sum[0], (512, 512), interpolation=cv2.INTER_LINEAR)
        #     # normalize logits to [0,1]          Question: or get average logits by logits/100?
        #     normalized_logits = (upsampled_logits - np.min(upsampled_logits)) / (np.max(upsampled_logits) - np.min(upsampled_logits))
        #     # set threshold for binary mask threshold that maximizes the true positive rate and minimized false positive rate
        #     # found through ROC curve over whole dataset where each image is reprompted 20 times
        #     thresh = 0.65
        #     thresholded_logits = np.zeros(normalized_logits.shape)
        #     for row in range(normalized_logits.shape[0]):
        #         for col in range(normalized_logits.shape[1]):
        #             if normalized_logits[row,col] > thresh:
        #                 thresholded_logits[row,col] = 254
        #             else: 
        #                 thresholded_logits[row,col] = 0
                        
        #     rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
        #     rgb_image[:, :, 0] = thresholded_logits  
        #     rgb_image[:, :, 1] = 0  
        #     rgb_image[:, :, 2] = 0  
            
        #     logit_mask = Image.fromarray(np.asarray(rgb_image))
        #     thresholded_logits_masks.append(logit_mask)
            
        #     # save image of path
        #     init_image_bgr = cv2.cvtColor(np.asarray(init_image), cv2.COLOR_RGB2BGR)
        #     thresholded_logits_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        #     overlay = cv2.addWeighted(init_image_bgr, 0.7,thresholded_logits_bgr, 0.8, 0)
        #     cv2.imwrite(f'{images_dir}/{file}', overlay)
        
        # compute metric
        metric, _ = compute_metric(sam_masks, files)
        print(metric)
        # save to text file
        with open('mean_iou.txt', 'a') as f:  # Open the file in append mode
            f.write(f"\n{metric}_{images_dir}\n")
            
            
        # """ roc curve for all images """
        # all_labels = []
        # all_scores = []
    
        # # get summed and scaled logits per 20 prompts
        # num_reprompts = 20
        # #images_dir = f'./path_sum_over_logits_{num_reprompts}'
        # images_dir = './ROC_curve_reprompts_80'
        # for file in tqdm(files):
        #     # get gt_masks
        #     gt_mask = get_GTmask_from_filename(file) # bool
        #     gt_mask_binary = gt_mask.astype(int)
        #     # prompt once
        #     multimask_output = True
        #     mask_input = None
        #     init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
        #     logits_sum = np.zeros((1,256,256))
        #     for i in range(num_reprompts):
        #         # prompt twice
        #         multimask_output = False
        #         mask_input = logits[np.argmax(scores), :, :]
        #         init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
        #         #sam_mask.save(f'./heatmap/multimask/{i}_{files[0]}')
        #         # add logits
        #         logits_sum += logits
        #     # upsample logits to match input image size
        #     upsampled_logits = cv2.resize(logits_sum[0], (512, 512), interpolation=cv2.INTER_LINEAR)
        #     # normalize logits to [0,1]          Question: or get average logits by logits/100?
        #     normalized_logits = (upsampled_logits - np.min(upsampled_logits)) / (np.max(upsampled_logits) - np.min(upsampled_logits))


        #     # Flatten the binary masks and labels
        #     labels = np.concatenate([mask.flatten() for mask in gt_mask_binary])
        #     scores = np.concatenate([mask.flatten() for mask in normalized_logits])
        #     all_labels.extend(labels)
        #     all_scores.extend(scores)

        # # Compute the false positive rate and true positive rate
        # fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        # # Pickle the arrays
        # with open(f'roc_data_reprompts_{num_reprompts}.pickle', 'wb') as f:
        #     pickle.dump((fpr, tpr, thresholds), f)
        # roc_auc = roc_auc_score(all_labels, all_scores)
        # print("roc_auc", roc_auc)
        
        
        # # calculate the g-mean for each threshold
        # gmeans = np.sqrt(tpr * (1-fpr))
        # ix = np.argmax(gmeans)
        # best_threshold = thresholds[ix]
        # print("best_threshold", best_threshold)
        
        # plt.plot(fpr, tpr)
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.savefig(f'./ROC_curve_{num_reprompts}.png')
        
    
        
        """ plot thresholds vs mean_iou for 20 reprompts per image """


        """ plot number of prompts vs mean iou of extracted mask """
        
        
        """ computes and saves metrics for whole dataset """
        # images_dir = './reprompt1_best_barams'
        # sam_masks = []
        # for file in tqdm(files):
        #     #init_image, mask_path, sam_mask, logits, coords, labels = pipeline(file)
            
        #     #prompt once
        #     multimask_output = True
        #     mask_input = None
        #     init_image, mask_path, sam_mask, scores, logits, coords, labels = pipeline(file, multimask_output, mask_input)
            
        #     # prompt twice
        #     multimask_output = False
        #     mask_input = logits[np.argmax(scores), :, :]
        #     init_image, mask_path, sam_mask, _, logits, coords, labels = pipeline(file, multimask_output, mask_input)

        
        #     sam_masks.append(sam_mask)
        #     combined_image = combine(init_image, sam_mask, coords=coords, labels=labels)
        #     # metric_for_one_pair, _ = compute_metric([sam_mask], [file])
        #     # save_metric_for_one_pair_sam(
        #     #     filename = file, 
        #     #     metric = metric_for_one_pair, 
        #     #     combined_image = combined_image, 
        #     #     images_dir = images_dir, 
        #     #     prompt = 'A bright photo of a road to walk on', 
        #     #     clipseg_mask = mask_path)
        #     combined_image.save(f'{images_dir}/{file}')

        # metric, _ = compute_metric(sam_masks, files)
        # print("metric: ", metric)

if __name__ == '__main__':
    main('sam')

# for i, mask in enumerate(masks):
    

# compute overall metric for all images in hike    
# metric = compute_metric(masks, files)
# # save to text file
# with open('mean_iou.txt', 'a') as f:  # Open the file in append mode
#     f.write(f"Metrics for mask_refined (SD with mask_path and seed) and ground truth pairs of hike dataset: {metric}\n")

# print(metric)



