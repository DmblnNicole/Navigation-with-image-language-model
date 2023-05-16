# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
import numpy as np
from PIL import Image
import sys
import re
import os

''' read masks from segments.ai '''
#np.set_printoptions(threshold=sys.maxsize)

# Initialize a SegmentsDataset from the release file
#client = SegmentsClient('24c8f65c43cd6b9d70f02b28b852b2676d543fa1')
#release = client.get_release('nicdmbln/youtube100', 'v0.1') # Alternatively: release = 'flowers-v1.0.json'
#dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to COCO panoptic format
#export_dataset(dataset, export_format='coco-panoptic')

''' convert masks to grayscale images GT_masks_youtube '''

for file in os.listdir('./segments/nicdmbln_youtube100/v0.1'):
    if re.search(r'coco-panoptic.png$', file):
        image_read_only = Image.open(f'./segments/nicdmbln_youtube100/v0.1/{file}')
        # convert to grayscale to avoid light blue color
        if image_read_only.mode != 'L':
            image_read_only = image_read_only.convert('L')
            
        image_read_only = np.asarray(image_read_only)
        image = np.copy(image_read_only)
        image[image != 0] = 255
        
        match = re.match(r"^([^_]*_[^_]*)", file)
        image_name = match.group(1)
        PIL_image = Image.fromarray(image).resize((512, 512))
        PIL_image.save(f'./GT_masks_youtube_resized/{image_name}.png')

''' convert predictions to grayscale masks '''

