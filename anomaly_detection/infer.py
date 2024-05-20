import torch
import pickle
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from perf_model import BiLSTM
from train import SeqDataset, PadCollate
import numpy as np
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
window_size = 10 # sequence window length

num_classes = 3
hidden_size = 128 # sequence embedding size
num_layers = 1
num_epochs = 30
batch_size = 32
attention_size = 16
model_dir = 'model'
time_dim = 1
bidirectional = True

def construct_vec(t2v, inputs):
    ret = np.zeros((len(inputs), embed_dim), np.float)
    for i in range(len(inputs)):
        eid = int(inputs[i])
        for j in range(embed_dim):
            ret[i][j] = t2v[eid][j]
    return torch.tensor(ret, dtype=torch.float)

def get_dataset(FN, s2v):
    # return list
    with open('data/' + FN, 'r') as f:
        for line in tqdm(f.readlines()):

            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            vecs = []
            for i in range(len(line)):
                eid = int(line[i]) + 1
                vecs.append(s2v[eid])
    return vecs 


def infer(args):
    encodingFN = args.encoding
    embed_dim = args.embeddim
    datasetName = args.dataset
    modelName = args.model
    tensortype = args.type
    indir = args.indir
    bi_flag = args.bi
    timeembedding=args.time
    timedim = args.timedim
    attnFlag = args.attn
    
    if tensortype == 'float32':
        tensortype = torch.float32
    else:
        tensortype = torch.double
    
    s2v = pickle.load(open(encodingFN, 'rb'))
    model = BiLSTM(batch_size, embed_dim, hidden_size, num_layers, num_classes,  bi_flag,
                   True, attnFlag, timedim, device)
    model.to(device)
    dataloader = DataLoader(
        SeqDataset(indir + '/my_' + datasetName + '_test_normal',
                   indir + '/my_' + datasetName + '_test_abnormal',
                   indir + '/my_' + datasetName + '_test_perf', encodingFN
                   ),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        collate_fn=PadCollate(dim=0,
                              typ=tensortype)
    )
    model.load_state_dict(torch.load(modelName))
    model.eval()

    start_time = time.time()
    label = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (seq, timedelta, label, leng) in tqdm(enumerate(dataloader)):
            y, output, seq_len = model(seq.to(device), timedelta.to(device), label.to(device), leng.to(device))
            _, preds = torch.max(output, 1)
            for i in range(len(preds)):
                y_true.append(y[i])
                y_pred.append(preds[i])

    y_true = [i.item() for i in y_true]
    y_pred = [i.item() for i in y_pred]
    f1 = f1_score(y_true, y_pred, average='macro' )
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')

    print('model:', modelName)
    print('dataset:', indir)
    print('classification report', classification_report(y_true, y_pred))
    print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(p*100, r*100, f1*100))
    print('Finished Predicting')
    elapsed_time = time.time() - start_time
    print('elapsed_time: {}'.format(elapsed_time))
