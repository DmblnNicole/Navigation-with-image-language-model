from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import utils

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
    
    mean_iou = utils.load('mean_iou')
    metric = mean_iou._compute(
        predictions=masks,
        references=gt_masks,
        num_labels=2, 
        ignore_index=255
    )
    return metric

        
