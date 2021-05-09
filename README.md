# DLProject

## Overview
The repository builds a code for anomaly detection in surveillance videos using CRNN model and CNN3D model with PyTorch.

## Requirements
* [PyTorch](http://pytorch.org/) (ver. 1.0+ required)
* Opencv
* Python3

## Preparation
* Download the dataset(videos) [here](https://webpages.uncc.edu/cchen62/dataset.html).

### Run
we have two models and two clients in this project: CNN3DClient and CRNNClient.

In Jupyter notebook or a seperate python file, import the client class and run.

For example:

```python
from Clients.crnn import CRNNClient
crnn = CRNNClient()
crnn.run()
```
