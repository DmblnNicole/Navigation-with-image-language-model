from setuptools import setup, find_packages

setup(
    name='navigate_with_image_language_model',
    version='1.1',
    python_requires=">=3.9",
    description='The package provides a pipeline that utilizes models like ClipSeg and StableDiffusion or ClipSeg and SegmentAnything to prompt an image for a path.',
    packages=find_packages(),
    install_requires=[
        'evaluate==0.4',
        'diffusers==0.12.1',
        'torch==2.0.0',
        'torchvision',
        'transformers==4.19.2',
        'tqdm',
        'seaborn==0.12.2',
        'scikit-learn==1.2.2',
        'opencv-python==4.7.0.72',
        'matplotlib==3.7.1',
        'accelerate==0.18'
    ],
)
