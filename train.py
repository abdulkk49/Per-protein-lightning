from os.path import join, exists, dirname, abspath, realpath
from os import system, chdir, getcwd, makedirs, listdir
import subprocess, numpy as np, time, sys, argparse, json, shutil
import pandas as pd
from numpy.random import choice
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
# print(dirname(abspath("__file__")))
sys.path.append(dirname(abspath("__file__")))
import pytorch_lightning as pl
import utils
from perproteindata import *
from perproteinet import *
from bert_embed import *


class PerProteinClassifier(pl.LightningModule):

    def __init__(self, params, loss_fn, bertModel, train_input_ids,\ 
                train_attention_mask, val_input_ids, val_attention_mask, metrics):
        super(PerProteinClassifier, self).__init__()
        self.params = params
        self.net = Net(self.params)
        self.loss = loss_fn
        self.bertModel = bertModel
        self.val_input_ids = val_input_ids
        self.train_input_ids = train_input_ids
        self.val_attention_mask = val_attention_mask
        self.train_attention_mask = train_attention_mask
        self.metrics = metrics

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
                e = self.bertModel(input_ids=input_ids[i:i+bs],attention_mask=attention_mask[i:i+bs])[0]
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


    def train_dataloader(self):
        print('Loading training data...')
        print('Train prefix : ', self.params.train_prefix)
        pwd = dirname(realpath("__file__"))
        direc = join(pwd, 'train')
        dataset = PerProteinDataset(join(direc,'loclabels.txt'), join(direc, 'memlabels.txt'), self.train_input_ids, self.train_attention_mask)
        loader = DataLoader(dataset, batch_size = self.params.batch_size, collate_fn = collate_fn, num_workers = params.num_workers)
        return loader
        
    def val_dataloader(self):
        print('Loading Validation data...')
        print('Valid prefix : ', self.params.valid_prefix)
        pwd = dirname(realpath("__file__"))
        direc = join(pwd, 'val')
        dataset = PerProteinDataset(join(direc,'loclabels.txt'), join(direc, 'memlabels.txt'), self.val_input_ids, self.val_attention_mask)
        loader = DataLoader(dataset, batch_size = self.params.batch_size, collate_fn = collate_fn, num_workers = params.num_workers)
        return loader
    
    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.net.parameters(), self.params.learning_rate)
        return self.optimizer

    def forward(self, batch):
        return self.net(batch)
      
    def training_step(self, batch, batch_idx):
        locoutput_batch, memoutput_batch = self(batch)

        locloss = loss_fn(locoutput_batch.cpu(), loclabels_batch)
        memloss = loss_fn(memoutput_batch.cpu(), memlabels_batch)

        loss = locloss + memloss

        locoutput_batch = locoutput_batch.data.cpu().numpy()
        memoutput_batch = memoutput_batch.data.cpu().numpy()
        memlabels_batch = memlabels_batch.data.numpy()
        loclabels_batch = loclabels_batch.data.numpy()
        # compute all metrics on this batch
        loc_acuracy = self.metrics['Loc_accuracy'](locoutput_batch, loclabels_batch), 
        mem_accuracy = self.metrics['Mem_accuracy'](memoutput_batch, memlabels_batch)}

   
        tensorboard_logs = {'train_loss': loss}
        return {
                'train_loss': loss,\
                'train_loc_accuracy': torch.from_numpy(np.array(loc_accuracy)),
                'train_mem_accuracy': torch.from_numpy(np.array(mem_accuracy)),
                'progress_bar': {'train_loss': loss, 'train_mem_accuracy': torch.from_numpy(np.array(mem_accuracy)),
                                'train_loc_accuracy': torch.from_numpy(np.array(loc_accuracy))},\
                'log': tensorboard_logs
              }

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['train_loss'] for x in outputs]).mean()
        # train_auc_mean = torch.stack([x['train_auc'] for x in outputs]).mean()
        train_loc_mean = torch.stack([x['train_loc_accuracy'] for x in outputs]).mean()
        train_mem_mean = torch.stack([x['train_mem_accuracy'] for x in outputs]).mean()

        
        results = {
            'log': {'avg_train_loss': train_loss_mean.item(), 'train_loc_acc': train_loc_mean.item(), 'train_mem_acc': train_mem_mean.item()},
            'progress_bar': {'avg_train_loss': train_loss_mean.item()}
        }
        return results

    def validation_step(self, data, batch_idx):
        locoutput_batch, memoutput_batch = self(batch)

        locloss = loss_fn(locoutput_batch.cpu(), loclabels_batch)
        memloss = loss_fn(memoutput_batch.cpu(), memlabels_batch)

        loss = locloss + memloss

        locoutput_batch = locoutput_batch.data.cpu().numpy()
        memoutput_batch = memoutput_batch.data.cpu().numpy()
        memlabels_batch = memlabels_batch.data.numpy()
        loclabels_batch = loclabels_batch.data.numpy()
        # compute all metrics on this batch
        loc_acuracy = self.metrics['Loc_accuracy'](locoutput_batch, loclabels_batch), 
        mem_accuracy = self.metrics['Mem_accuracy'](memoutput_batch, memlabels_batch)}

   
        tensorboard_logs = {'val_loss': loss}
        return{
                'val_loss': loss,\
                'val_loc_accuracy': torch.from_numpy(np.array(loc_accuracy)),
                'val_mem_accuracy': torch.from_numpy(np.array(mem_accuracy)),
                'progress_bar': {'val_loss': loss, 'val_mem_accuracy': torch.from_numpy(np.array(mem_accuracy)),
                                'val_loc_accuracy': torch.from_numpy(np.array(loc_accuracy))},\
                'log': tensorboard_logs
              }

    def validation_epoch_end(self, outputs): 
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        # train_auc_mean = torch.stack([x['train_auc'] for x in outputs]).mean()
        val_loc_mean = torch.stack([x['val_loc_accuracy'] for x in outputs]).mean()
        val_mem_mean = torch.stack([x['val_mem_accuracy'] for x in outputs]).mean()
    
        results = {
            'log': {'avg_val_loss': train_loss_mean.item(), 'val_loc_acc': train_loc_mean.item(), 'val_mem_acc': train_mem_mean.item()},
            'progress_bar': {'avg_val_loss': train_loss_mean.item()}
        }
        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./Embeddings', help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='./trial', help="Directory containing params.json")
    parser.add_argument('--batch', default=128, help="Batch Size for Training")
    parser.add_argument('--num_workers', default=1, help="Num workers for Training")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)

    # Load parameters
    json_path = os.path.join(pwd, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()
    params.batch_size = int(args.batch)
    params.num_workers = int(args.num_workers)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    print("Params: " ,params.__dict__)

    # Invoke Bert Model
    bertModel, tokenizer = embedModel()
    bertModel = bertModel.eval()
    
    #Load sequences
    sequences_Example =[]
    count = 0
    with open("./train/sequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total training data points(Clean): ", str(count))
    
    #Replace "UZOB" with "X"
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Tokenizing input sequences
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    train_input_ids = torch.tensor(ids['input_ids']).to(device)
    train_attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    sequences_Example =[]
    count = 0
    with open("./val/sequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total validation data points(Clean): ", str(count))
    
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    val_input_ids = torch.tensor(ids['input_ids']).to(device)
    val_attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    del sequences_Example

    loss_fn = loss_fn
    metrics = metrics

    model = PerProteinClassifier(params, loss_fn, bertModel, train_input_ids,\
                                 train_attention_mask, val_input_ids, val_attention_mask, metrics)
    outdir = join(pwd, 'trial')
    trainer = pl.Trainer(default_root_dir = outdir, max_epochs = 10) 
    trainer.fit(model)
