# material-scripts-asr-segmentation

This is the SCRIPTS' ASR Segmentation component for lt, bg, and fa.

## Requirements
- pytorch 1.7.0
- sacremoses
- torchtext 0.8.0
- tokenizers
- pytorch-lightning 0.9.0

## Code
The main entry file is main.py, using Pytorch-Lightning framework.

To run the code, please check the scripts folder. The entry point is `scripts/segment.sh`, which will call various scripts to preprocess, running the code, an then post-processing.

It expects an input to the asr directory to perform segmentation and an output directory to save.


## Docker

First, download the models to `models` directory.


To build a Docker image
```
$ docker build -t asr_segmentation .
```

To run the image
```
$ docker run --gpus all -v [ASR_PATH]:/input -v [OUTPUT_PATH]:/output asr_segmentation:latest [LANGUAGE]
```
