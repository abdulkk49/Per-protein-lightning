"""Evaluates the model"""

import argparse
import logging
import os, sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import utils
# from evaluate import evaluate
from os.path import join, exists, dirname, abspath, realpath

sys.path.append(dirname(abspath("__file__")))

from models.data_loader import *
from models.net import *
from transformers import BertModel, BertTokenizer
import re
import os
import requests, h5py
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./Embeddings',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='./trial',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--batch', default=128,
                    help="Batch Size for Training")
parser.add_argument('--num_workers', default=1,
                    help="Num workers for Training")

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



bertModel, tokenizer = embedModel()
bertModel = bertModel.eval()

def collate_fn(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    # print(input_ids.shape, attention_mask.shape)
    embedding = torch.zeros((input_ids.shape[0],1024), dtype = torch.float32).to(device)
    i = 0
    bs = 32
    x = 0
    with torch.no_grad():
        while i + bs <= len(input_ids):
            e = bertModel(input_ids=input_ids[i:i+bs],attention_mask=attention_mask[i:i+bs])[0]
            # print(e.shape)
            for seq_num in range(len(e)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = e[seq_num, 1:seq_len-1, :]
                seq_emd = torch.mean(seq_emd, -2)
                # print(seq_emd.shape)
                embedding[x,:] = seq_emd
                x += 1
        
            i += bs
            print(e.shape, i)

    with torch.no_grad(): 
    	if i != len(input_ids):
            #Final batch < 32
            e = bertModel(input_ids=input_ids[i:len(input_ids)],attention_mask=attention_mask[i:len(input_ids)])[0]
            # print(e.shape)
            for seq_num in range(len(e)):
                seq_len = (attention_mask[seq_num] == 1).sum()
                seq_emd = e[seq_num, 1:seq_len-1, :]
                seq_emd = torch.mean(seq_emd, -2)
                # print(seq_emd.shape)
                embedding[x,:] = seq_emd
                x += 1
            print(e.shape, i)

    loclabel = torch.stack([item[2] for item in batch])
    memlabel = torch.stack([item[3] for item in batch])
    print(loclabel.shape, memlabel.shape)
    return embedding, loclabel, memlabel

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    for val_batch, loclabels_batch, memlabels_batch in dataloader:
            # move to GPU if available
            # if params.cuda:
            #     val_batch, q8labels_batch = val_batch.cuda(non_blocking=True), q8labels_batch.cuda(non_blocking=True)
            # if params.cuda:
            #     q3labels_batch = q3labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            # N x 3 x 1632, N x 6 x 1632
            locoutput_batch, memoutput_batch = model(val_batch)


            locloss = loss_fn(locoutput_batch.cpu(), loclabels_batch)
            memloss = loss_fn(memoutput_batch.cpu(), memlabels_batch)

            loss = locloss + memloss
            
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            locoutput_batch = locoutput_batch.data.cpu().numpy()
            memoutput_batch = memoutput_batch.data.cpu().numpy()
            memlabels_batch = memlabels_batch.data.numpy()
            loclabels_batch = loclabels_batch.data.numpy()

            # mask shape = N x 1632
            # compute all metrics on this batch
            summary_batch = {'val_locaccuracy': metrics['Loc_accuracy'](locoutput_batch, loclabels_batch), 'val_memaccuracy': metrics['Mem_accuracy'](memoutput_batch, memlabels_batch)}
            summary_batch['val_loss'] = loss.item()
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Validation metrics: " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters from json file
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    args = parser.parse_args()
    json_path = os.path.join(pwd, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    params.batch_size = int(args.batch)
    params.num_workers = int(args.num_workers)
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    print("Params: " ,params.__dict__)

    #Load sequences
    sequences_Example =[]
    count = 0
    with open("./test/sequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total testing data points(Clean): ", str(count))
    
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Tokenizing input sequences
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    test_input_ids = torch.tensor(ids['input_ids']).to(device)
    test_attention_mask = torch.tensor(ids['attention_mask']).to(device)
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    test_dl = fetch_dataloader('test', 'loclabels.txt', 'memlabels.txt', test_input_ids, test_attention_mask, params, collate_fn)
    
    del test_input_ids
    del test_attention_mask
    logging.info("- done.")

    # Define the model
    model = ProteinNet(params)
    if params.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = loss_fn
    metrics = metrics

    logging.info("Starting evaluation..")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
