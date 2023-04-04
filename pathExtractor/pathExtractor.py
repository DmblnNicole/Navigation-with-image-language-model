import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


class PathExtractor:
    def __init__(self, path_original_image, path_result_image):
        self.path_original_image = path_original_image
        self.path_result_image = path_result_image
        self.thresh = 150
        self.thresh_ = 250
    
    def comparePixel(self):
        original_image = cv2.imread(self.path_original_image)
        result_image = cv2.imread(self.path_result_image)
        for y in range(len(original_image)):
            for x in range(len(original_image)):
                if abs(original_image[y][x][0]-result_image[y][x][0]) < self.thresh and abs(original_image[y][x][1]-result_image[y][x][1]) < self.thresh and abs(original_image[y][x][2]-result_image[y][x][2]) < self.thresh:
                    # original and result are the same
                    result_image[y][x] = [0,0,0] # black
                else:
                    # original and result are different
                    result_image[y][x] = [255,0,0] # red
                
        
        return result_image


# Define paths to original and result image
path_original_image = './input/img11_inpainting_input.png'
path_result_image = './input/img11_inpainting_output.png'
image_name = "img11_inpainting_output"

# Compare images pixelwise
pathExtractor = PathExtractor(path_original_image, path_result_image)
resultHeatmap = pathExtractor.comparePixel()
#cv2.imshow('heatmap', resultHeatmap)
resultHeatmap = Image.fromarray(np.uint8(resultHeatmap)).convert('RGB')
resultHeatmap.save(f'./results/resultHeatmap_{image_name}.png')
cv2.waitKey(20000)
cv2.destroyAllWindows()