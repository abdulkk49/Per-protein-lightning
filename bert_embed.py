import torch, numpy as np
from transformers import BertModel, BertTokenizer
import re
import os
import requests, h5py
from tqdm.auto import tqdm
from os.path import join, exists, dirname, abspath, realpath

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def embedModel():
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    #Pretrained Model files
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    #Setting folder paths
    downloadFolderPath = 'models/ProtBert/'
    modelFolderPath = downloadFolderPath

    #Setting file paths
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    #Creading model directory
    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    #Downloading pretrained model
    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)
    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)
    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    #Initializing Tokenizer, Model
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, tokenizer

