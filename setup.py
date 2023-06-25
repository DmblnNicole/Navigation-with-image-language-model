from setuptools import setup, find_packages

setup(
    name='navigation_with_image_language_model',
    version='1.0',
    description='ClipSeg and StableDiffusion or ClipSeg and SegmentAnything are utilized in a pipeline to prompt an image for a path',
    packages=find_packages(),
    install_requires=[
        'accelerate==0.18.0',
        'aiohttp==3.8.4',
        'aiosignal==1.3.1',
        'asttokens==2.2.1',
        'async-timeout==4.0.2',
        'attrs==22.2.0',
        'backcall==0.2.0',
        'backports.functools-lru-cache==1.6.4',
        'certifi==2022.12.7',
        'charset-normalizer==3.1.0',
        'clip==1.0',
        'clipseg==0.0.1',
        'cmake==3.26.1',
        'coloredlogs==15.0.1',
        'contourpy==1.0.7',
        'cycler==0.11.0',
        'datasets==2.11.0',
        'debugpy==1.5.1',
        'decorator==5.1.1',
        'diffusers==0.12.1',
        'dill==0.3.6',
        'entrypoints==0.4',
        'evaluate==0.4.0',
        'executing==1.2.0',
        'filelock==3.10.7',
        'flatbuffers==23.3.3',
        'fonttools==4.39.3',
        'frozenlist==1.3.3',
        'fsspec==2023.4.0',
        'ftfy==6.1.1',
        'huggingface-hub==0.13.3',
        'humanfriendly==10.0',
        'idna==3.4',
        'imageio==2.27.0',
        'importlib-metadata==6.1.0',
        'importlib-resources==5.12.0',
        'ipykernel==6.15.0',
        'ipython==8.12.0',
        'isort==5.12.0',
        'jedi==0.18.2',
        'Jinja2==3.1.2',
        'joblib==1.2.0',
        'jupyter-client==7.0.6',
        'jupyter_core==5.3.0',
        'kiwisolver==1.4.4',
        'lazy_loader==0.2',
        'lit==16.0.0',
        'MarkupSafe==2.1.2',
        'matplotlib==3.7.1',
        'matplotlib-inline==0.1.6',
        'mpmath==1.3.0',
        'multidict==6.0.4',
        'multiprocess==0.70.14',
        'nest-asyncio==1.5.6',
        'networkx==3.0',
        'numpy==1.24.2',
        'nvidia-cublas-cu11==11.10.3.66',
        'nvidia-cuda-cupti-cu11==11.7.101',
        'nvidia-cuda-nvrtc-cu11==11.7.99',
        'nvidia-cuda-runtime-cu11==11.7.99',
        'nvidia-cudnn-cu11==8.5.0.96',
        'nvidia-cufft-cu11==10.9.0.58',
        'nvidia-curand-cu11==10.2.10.91',
        'nvidia-cusolver-cu11==11.4.0.1',
        'nvidia-cusparse-cu11==11.7.4.91',
        'nvidia-nccl-cu11==2.14.3',
        'nvidia-nvtx-cu11==11.7.91',
        'onnx==1.13.1',
        'onnxruntime==1.14.1',
        'opencv-python==4.7.0.72',
        'packaging==23.0',
        'pandas==2.0.0',
        'parso==0.8.3',
    ],
)
