# Navigation with image language model

This project presents a pipeline for robot navigation that uses the [ClipSeg](https://arxiv.org/pdf/2112.10003.pdf) and [Segment Anything](https://arxiv.org/pdf/2304.02643.pdf) models to generate masks for traversable paths in images. This approach is well suited for paths that have a high contrast and are already visible. To retrieve a new path, a pipeline prompting ClipSeg and Stable Diffusion is implemented.

### ClipSeg and Segment Anything
Original image          |  Final mask
:-------------------------:|:-------------------------:
![orig_youtube](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/e2dfbc2d-6b7b-4bae-9d0a-385536ee30aa)  |  ![SAM_youtube100](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/a9ddc143-6634-490c-8c26-41313c7cc3cd)

### ClipSeg and Stable Diffusion
Original image          |  Final mask
:-------------------------:|:-------------------------:
![orig_hike](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/3df134b8-c2cc-48c9-91cd-e5e32a6b5db8) |  ![sd_final_mask_hike](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/85cc2749-e326-4653-a8a0-672d0de74649)

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
![wide_angle_camera_front_1677756688_627165488](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/d723579e-1f40-496f-a3d9-019c1a834ffd)|![wide_angle_camera_front_1677756728_599394188](https://github.com/DmblnNicole/Navigation-with-image-language-model/assets/75450536/b4f403fc-6562-48a7-914d-1031507d70c6)
