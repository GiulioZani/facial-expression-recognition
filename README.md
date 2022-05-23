# Computer Vision: Emotion Recognition

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
python -m dl.preprocess --dataset_path=dataset/affectnet.zip --output_path=dataset/affectnet
```

Test the model:
```
python -m dlpm test resnet
```


