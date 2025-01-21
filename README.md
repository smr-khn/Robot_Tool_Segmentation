# Robot Tool Segmentation using Branch Aggregation Attention Network
This repo uses an adaptation of the BAAnet from the paper Branch Aggregation Attention Network for
Robotic Surgical Instrument Segmentation by W. Shen et al. The model was trained and tested on the Endovis 2018 dataset.

## Contents
```EncoderModule.py```: Encoder module for BAAnet that leverages mobilenetv2.

```DecoderModule.py```: Block attention fusion module that uses dual branch attention transformer architecture.

```BBAModule.py```: Branch balance aggregation module takes the multi-level feature output of the encoder and upsamples/combines.

```BAANet.py```: Puts together the many modules and creates a full branch aggregation attention network

```TrainingUtils.py```: Helper functions for training and testing model

## How to Use
In order to use or test the modules, you can import the classes and functions from above or use the two jupyter notebooks provided. ```BAAnet.ipynb`` is a notebook that runs better locally while ```Test_MLDL.ipynb``` runs better on Google Colab if that is your preferred method. Be wary of wear the data is located and follow the format used in the repo/notebook.
