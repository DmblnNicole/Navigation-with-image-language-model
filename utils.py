from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import evaluate

def bitmap2image(semantic_bitmap) -> Image:
    A = np.asarray(semantic_bitmap, dtype=np.uint8)
    image = np.zeros((A.shape[0], A.shape[1], 3), dtype=np.uint8)
    image[:,:,0] = A*255
    image[:,:,1] = A*255
    image[:,:,2] = A*255
    image = Image.fromarray(image, 'RGB')
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def combine(image : Image, semantic_mask : Image, coords=None, labels=None) -> Image:
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.imshow(image)
    mask = show_mask(image2bitmap(semantic_mask, dtype=np.bool_), plt)
    ax.imshow(mask)
    ax.axis('on')
    if isinstance(coords, np.ndarray):
        # print(coords)
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=200, edgecolor='white', linewidth=1.25)   
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi() 
    img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
    img = Image.fromarray(img, 'RGB')
    return img    

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
    # plt.suptitle(f"mean_iou: {metric['mean_iou']}\n" 
    #             f"mean_accuracy: {metric['mean_accuracy']}\n"
    #             f"per_category_iou: {metric['per_category_iou']}\n"
    #             f"per_category_accuracy: {metric['per_category_accuracy']}")
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