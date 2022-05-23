# Computer Vision: Emotion Recognition

The goal of this project is to build a model that can recognize the emotion of a face. The model will be trained on a dataset of images of faces and their emotion. The model will be able to recognize the emotion of a face based on the image. 

We have used pytorch, pytorch lightning, and pytorch vision.

The file containig the model used, resnet can be found in the `dl/base_torch_modules/resnetmodel.py`. The pretrained model is contained in the `dl/models/resnet/checkpoint.ckpt`.

## Installation
Make sure you have at least python 3.9 installed.
If the `dlpm` folder is empty, you need to clone the repository.

```bash
git clone https://github.com/GiulioZani/dlpm
```
Then install the dependecies.
```bash
pip install -r dlpm/requirements.txt
```
```bash
pip install -r dl/requirements.txt
```
`dlpm` is a framework for running and testing deep learning models. The code relevant to this specific task is contained in the `dl` folder.

## Download the dataset
The dataset can be downloaded form [kaggle.](https://www.kaggle.com/datasets/tom99763/affectnethq) The size is 9GB.

```
python -m dl.preprocess --dataset_path=<unzipped folder containing data> --destination_folder=<preprocessed location>
```


## Test the model
First you need to edit the `dl/models/resnet/default_parameters.json` file and set the `data_location` parameter to the location of the preprocessed dataset. Then run:
```
python -m dlpm test resnet
```

To classify webcam videos, run:
```
python -m dlpm exec resnet
```

