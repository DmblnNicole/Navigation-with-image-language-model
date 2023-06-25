import evaluate
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torchvision import transforms
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
""" general utils """

def image2bitmap(image : Image, dtype=np.bool_) -> np.array:
    image = np.asarray(image, dtype=dtype)
    bitmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.bool_)
    bitmap = image[:,:,0]
    return bitmap

def bitmap2image(semantic_bitmap) -> Image:
    A = np.asarray(semantic_bitmap, dtype=np.uint8)
    image = np.zeros((A.shape[0], A.shape[1], 3), dtype=np.uint8)
    image[:,:,0] = A*255
    image[:,:,1] = A*255
    image[:,:,2] = A*255
    image = Image.fromarray(image, 'RGB')
    return image

""" evaluation utils """

def get_GTmask_from_filename(GT_dir:str, filename: str):
    resize = transforms.Resize((512,512))
    a = Image.open(GT_dir+'/'+filename)
    gt_mask_array = np.asarray(resize(a))
    gt_mask_bool = gt_mask_array.astype(bool)
    # make sure GT has shape (512,512) and not (512,512,3)
    if (gt_mask_bool.shape != (512,512)):
        gt_mask_bool = gt_mask_bool[:, :, 0]
    return gt_mask_bool


def compute_metric(GT_dir: str, masks, files):
    masks = [image2bitmap(mask) for mask in masks]
    gt_masks = []
    for i, file in enumerate(files):
        gt_masks.append(get_GTmask_from_filename(GT_dir, file))
    mean_iou = evaluate.load('mean_iou')
    metric = mean_iou._compute(
        predictions=masks,
        references=gt_masks,
        num_labels=2,
        ignore_index=255
    )
    return metric, gt_masks

def getMetric(metric):
    mean_iou = "{:.2f}".format(metric['mean_iou'])
    mean_accuracy = "{:.2f}".format(metric['mean_accuracy'])
    per_category_iou = ["{:.2f}".format(m) for m in metric['per_category_iou']]
    per_category_accuracy = ["{:.2f}".format(m) for m in metric['per_category_accuracy']]
    overall_accuracy = "{:.2f}".format(metric['overall_accuracy'])
    return [mean_iou, mean_accuracy, per_category_iou, per_category_accuracy, overall_accuracy]

        
""" visualization utils """

def combine(image : Image, clip_mask, sam_mask=None, coords=None, labels=None) -> Image:
    fig = Figure(figsize=(5.12, 5.12), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.imshow(image)
    clip_mask = show_mask(image2bitmap(clip_mask, dtype=np.bool_), plt)
    ax.imshow(clip_mask)
    if isinstance(sam_mask, Image.Image):
        sam_mask = show_mask(image2bitmap(sam_mask, dtype=np.bool_), plt, random_color=True)
        ax.imshow(sam_mask)
    if isinstance(coords, np.ndarray):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=50, linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=50, linewidth=1.25)   
    ax.axis('off')
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi() 
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
    img = Image.fromarray(img, 'RGB')
    return img

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
    plt.close() 
        
def save_metric_for_one_pair_with_SD_output(filename, init_image, SD_output, mask, metric, title, images_dir, prompt):
    gt_mask = get_GTmask_from_filename(filename)
    mask = image2bitmap(mask)

    plt.figure(figsize=(12,12))
    plt.suptitle(f"mean_iou: {getMetric(metric)[0]}  " 
                f"mean_acc: {getMetric(metric)[1]}  "
                f"per_category_iou: {getMetric(metric)[2]}  "
                f"per_category_acc: {getMetric(metric)[3]}\n\n"
                f"CligSeg prompt: {prompt}")
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
    plt.close() 
    
def save_metric_for_one_pair_sam(filename, metric, combined_image, images_dir, prompt, clipseg_mask):
    gt_mask = get_GTmask_from_filename(filename)
        
    plt.figure(figsize=(16, 6), tight_layout=True)
    plt.suptitle(f"mean_iou: {getMetric(metric)[0]}  "
                 f"mean_acc: {getMetric(metric)[1]}  "
                 f"overall_accuracy: {getMetric(metric)[4]}  "
                 f"per_category_iou: {getMetric(metric)[2]}  "
                 f"per_category_acc: {getMetric(metric)[3]}\n\n")
    
    plt.subplot(1, 3, 1)
    plt.title('ClipSeg Mask')
    plt.imshow(clipseg_mask)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('ClipSeg + SAM')
    plt.imshow(combined_image)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.imshow(gt_mask)
    plt.axis('off')
    
    plt.savefig(f"{images_dir}/{filename}")
    plt.close()  
    

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

