# Navigation with image language model

This project presents a pipeline for robot navigation that uses the [ClipSeg](https://arxiv.org/pdf/2112.10003.pdf) and [Segment Anything](https://arxiv.org/pdf/2304.02643.pdf) models to generate masks for traversable paths in images. This approach is well suited for paths that have a high contrast and are already visible. To retrieve a new path, a pipeline prompting ClipSeg and Stable Diffusion is implemented.

### ClipSeg and Segment Anything
Original image          |  Final mask
:-------------------------:|:-------------------------:
![orig_youtube](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/530166e2-d0d0-4143-bc31-f9889bd933a0)  |  ![img_000238](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/604d702c-db51-41b0-a52d-ef8ad8939702)


### ClipSeg and Stable Diffusion
Original image          |  Final mask
:-------------------------:|:-------------------------:
![orig_hike](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/de5ae4d9-0daa-4235-af16-448d0810cddc) | ![sd_final_mask_hike](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/7188d901-e523-4b54-ba49-8cdbb3a7f69d)



## Installation

1. Clone the repository locally and pip install `navigate-with-image-language-model` with:
   
  ```
  git clone https://github.com/DmblnNicole/Navigation-with-image-language-model.git
  pip install -e .
 ```

2. Install Dependecies

```
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/segment-anything.git
```

3. Download the checkpoint for Segment Anything model type vit_h here: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and save it in the root folder of the repository.

## Getting started

The file `pipeline/eval.py` runs the pipeline and contains all adjustable information like textprompts, paths to image data and the model types.

- Choose your model type and specify if output masks should be visualized.
```
if __name__ == '__main__':
    main('sam', visualize=False)
```
A new folder called ` output ` will save the masks if `visualize==True`.

### Optional

Change text prompts and upload your own dataset. 

- Upload image data and specify path
```
data_path='../data/images/hike/edge'
```
- Upload ground truth masks and specify path
```
GT_dir = '../data/GT/GT_hike'
```
- Choose your text prompt
```
word_mask='A bright photo of a road to walk on'
```
## Experimental Results
The pipeline, comprising ClipSeg and Segment Anything, was evaluated on a dataset extracted from YouTube videos as shown above. This dataset consists of images with visible paths and high contrast. While the primary objective is to segment already visible paths with ClipSeg and Segment Anything, the method also produces results for images of forest terrain where no clear path is visible.

Final Mask         |  Final mask
:-------------------------:|:-------------------------:
![wide_angle_camera_front_1677756688_627165488](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/8cfd3f22-d62b-4416-8c85-a08fdf176640)|![wide_angle_camera_front_1677756728_599394188](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/09458801-efea-406a-a087-099c01349862)

