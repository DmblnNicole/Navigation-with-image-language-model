import cv2
import evaluate
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import timeit
from PIL import Image
from ClipSegStableDiffusionPipeline import ClipSegStableDiffusionPipeline
    

def image2bitmap(image : Image, dtype=np.bool_) -> np.array:
    image = np.asarray(image, dtype=dtype)
    bitmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool_)
    bitmap = image[:,:,0]
    return bitmap

def get_GTmask_from_filename(filename: str):
    resize = transforms.Resize((512,512))
    a = Image.open('GT_masks/'+filename)
    gt_mask = image2bitmap(resize(a))
    return gt_mask

def compute_metric(masks, files):
    masks = [image2bitmap(mask) for mask in masks]
    gt_masks = []
    for i, file in enumerate(files):
        gt_masks.append(get_GTmask_from_filename(file))
    
    mean_iou = evaluate.load('mean_iou')
    metric = mean_iou._compute(
        predictions=masks,
        references=gt_masks,
        num_labels=2, 
        ignore_index=255
    )
    return metric

        
def save_metric_for_one_pair(filename, init_image, mask, metric, title, images_dir):
    gt_mask = get_GTmask_from_filename(filename)
    mask = image2bitmap(mask)

    plt.figure(figsize=(13,6))
    plt.suptitle(f"mean_iou: {metric['mean_iou']}\n" 
                f"mean_accuracy: {metric['mean_accuracy']}\n"
                f"per_category_iou: {metric['per_category_iou']}\n"
                f"per_category_accuracy: {metric['per_category_accuracy']}")    
    plt.subplot(1,3,1)
    plt.title('original_image')
    plt.imshow(init_image)
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.title(title)
    plt.imshow(np.asarray(mask))
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.title('ground_truth')
    plt.imshow(gt_mask)
    plt.axis('off')
    plt.savefig(f"{images_dir}/{filename}")
    plt.close() # Close the figure after saving

def save_metric_for_one_pair_with_SD_output(filename, init_image, SD_output, mask, metric, title, images_dir):
    gt_mask = get_GTmask_from_filename(filename)
    mask = image2bitmap(mask)

    plt.figure(figsize=(12,12))
    plt.suptitle(f"mean_iou: {metric['mean_iou']}\n" 
                f"mean_accuracy: {metric['mean_accuracy']}\n"
                f"per_category_iou: {metric['per_category_iou']}\n"
                f"per_category_accuracy: {metric['per_category_accuracy']}")
    plt.subplot(2,2,1)
    plt.title('original_image')
    plt.imshow(init_image)
    plt.axis('off')
    plt.subplot(2,2,2)
    plt.title('stable_diffusion_inpainting')
    plt.imshow(SD_output)
    plt.axis('off')
    plt.subplot(2,2,3)
    plt.title('ground_truth')
    plt.imshow(gt_mask)
    plt.axis('off')
    plt.subplot(2,2,4)
    plt.title(title)
    plt.imshow(np.asarray(mask))
    plt.axis('off')
    plt.savefig(f"{images_dir}/{filename}")
    plt.close() # Close the figure after saving

def measure_time_elapsed():
    pass

def show_max_IOU_masks():
    pass

def show_min_IOU_masks():
    pass
    
    

pipeline = ClipSegStableDiffusionPipeline()
files = pipeline.loadData()
images_dir = './metrics/youtube/mask_refined_SD_with_mask_path'
masks = []

#compute metric for single mask/ground_truth pairs
for i, file in enumerate(files):
    mask_path, mask_obstacle, mask_combined, mask_refined, stable_diffusion_output, init_image = pipeline.forwardPass(file)
    metric_for_one_pair = compute_metric([mask_refined], [file])
    save_metric_for_one_pair_with_SD_output(file, init_image, stable_diffusion_output, mask_refined, metric_for_one_pair, title='mask_refined', images_dir=images_dir)
    masks.append(mask_refined)
    
# compute overall metric for all images in hike    
metric = compute_metric(masks, files)
# save to text file
with open('mean_iou.txt', 'a') as f:  # Open the file in append mode
    f.write(f"Metrics for mask_refined (SD with mask_path and seed) and ground truth pairs of hike dataset: {metric}\n")

print(metric)



