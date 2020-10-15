#######################################################
#This script is for evaluating for the task of SLR    #
#######################################################

import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import datetime as dt
import _pickle as pickle
from collections import OrderedDict

from transformer_slt import make_model as TRANSFORMER
from dataloader_slt import loader
from utils import path_data, Batch, greedy_decode

#Progress bar to visualize training progress
import progressbar

import matplotlib.pyplot as plt

#Evaluation metrics
from bleu import compute_bleu
from rouge import rouge
from beam_search import beam_decode
#from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
#https://pypi.org/project/py-rouge/
#import rouge

#Produce the translation using greedy decoding or beam search
def decoding(model, src, batch, hand_regions=None, start_symbol=1, max_len=20, device='cuda', method='greedy', n_beam=8):

    src = Variable(src)

    if(method=='greedy'):
        translations = greedy_decode(model, src, hand_regions, batch.rel_mask, batch.src_mask, max_len=max_len, start_symbol=start_symbol, n_devices=1)

    elif(method=='beam'):
        translations = beam_decode(model, src, hand_regions, batch.rel_mask, batch.src_mask, max_len=max_len, start_symbol=start_symbol, n_beam=n_beam)

    else:
        print("Decoding method is not supported !")
        quit(0)

    return translations

###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Evaluation')


parser.add_argument('--data', type=str, default=os.path.join('data','PHOENIX-2014-T'),
                    help='location of the test data corpus')

parser.add_argument('--model_path', type=str, default=os.path.join("EXPERIMENTATIONS"),
                    help='location of the test data corpus')

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slt_lookup.txt'),
                    help='location of the words lookup table')

parser.add_argument('--lookup_table_gloss', type=str, default=os.path.join('data','slt_lookup_gloss.txt'),
                    help='location of the words gloss lookup table')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images. NOTE: use same image size as the training or else you get worse results.')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=4,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='size of one minibatch')

parser.add_argument('--save', action='store_true',
                    help='save the results of the evaluation')

parser.add_argument('--hand_query', type=bool, default=False,
                    help='Set hand cropped image as a query for transformer network.')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/resnet')

parser.add_argument('--decoding', type=str, default='greedy',
                    help='Decoding method (greedy/beam).')

parser.add_argument('--n_beam', type=int, default=4,
                    help='Beam width when using bean search for decoding.')

parser.add_argument('--decoding_length', type=int, default=20,
                    help='Set the maximum decoding length. NOTE: to get the same results as when training you need to use the same max_len')

parser.add_argument('--rel_window', type=int, default=None)

parser.add_argument('--bleu', action='store_true',
                    help='Use bleu for evaluation.')

parser.add_argument('--rouge', action='store_true',
                    help='Use rouge for evaluation.')

#----------------------------------------------------------------------------------------


#Same seed for reproducibility)
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#Save folder with the date
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
print ("Start Time: "+start_date)

args = parser.parse_args()

#Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#experiment_path = PureWindowsPath('EXPERIMENTATIONS\\' + start_date)
save_path = os.path.join('EVALUATION', start_date)

# Creates an experimental directory and dumps all the args to a text file
if(args.save):
    if(os.path.exists(save_path)):
        print('Evaluation already exists..')
    else:
        os.makedirs(save_path)

    print ("\nPutting log in EVALUATION/%s"%start_date)

    #Dump all configurations/hyperparameters in txt
    with open (os.path.join(save_path,'eval_config.txt'), 'w') as f:
        f.write('Experimentation done at: '+ str(start_date)+' with current configurations:\n')
        for arg in vars(args):
            f.write(arg+' : '+str(getattr(args, arg))+'\n')

#-------------------------------------------------------------------------------

#Run on GPU
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
else:
#Run on CPU
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")


train_path, valid_path, test_path = path_data(data_path=args.data, task='SLT', hand_query=args.hand_query)


#No data augmentation for test data
test_dataloader, test_size = loader(csv_file=test_path[1],
                root_dir=test_path[0],
                lookup_gloss=args.lookup_table_gloss,
                lookup=args.lookup_table,
                rescale = args.rescale,
                augmentation = False,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                show_sample = args.show_sample,
                istrain=False,
                hand_dir=test_path[2]
                )

#No data augmentation for test data
valid_dataloader, valid_size = loader(csv_file=valid_path[1],
                root_dir=valid_path[0],
                lookup=args.lookup_table,
                lookup_gloss=args.lookup_table_gloss,
                rescale = args.rescale,
                augmentation = False,
                batch_size = args.batch_size,
                num_workers = args.num_workers,
                show_sample = args.show_sample,
                istrain=False,
                hand_dir=valid_path[2]
                )

print('Test dataset size: '+str(test_size))
print('Valid dataset size: '+str(valid_size))

#Retrieve size of target vocab
with open(args.lookup_table, 'rb') as pickle_file:
   vocab = pickle.load(pickle_file)

vocab_size = len(vocab)

#Switch keys and values of vocab to easily look for words
vocab = {y:x for x,y in vocab.items()}

print('vocabulary size:' + str(vocab_size))

#Load entire model w/ weights
#NOTE: evaluate on single device
model = torch.load(args.model_path, map_location=device)

#Load params
#model.load_state_dict(n)

model = model.to(device)
print("Model successfully loaded")

model.eval()   # Set model to evaluate mode
print ("Evaluating..")

start_time = time.time()

#Loop through test and val sets
dataloaders = [valid_dataloader, test_dataloader]
sizes = [valid_size, test_size]
dataset = ['Validation Set', 'Test Set']

for d in range(len(sizes)):

    dataloader = dataloaders[d]
    size = sizes[d]
    print(dataset[d])

    #For progress bar
    bar = progressbar.ProgressBar(maxval=size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i = 0
    count = 0

    #Save translation and reference sentences
    translation_corpus = []
    reference_corpus = []
    references_corpus = []

    #Loop over minibatches
    for step, (x, x_lengths, y, y_lengths, gloss, gloss_lengths) in enumerate(dataloader):

        #Update progress bar with every iter
        i += len(x)
        bar.update(i)

        y = torch.from_numpy(y).to(device)
        x = x.to(device)

        batch = Batch(x_lengths, y_lengths, y, DEVICE=device, emb_type='2d', fixed_padding=None, rel_window=args.rel_window)

        with torch.no_grad():
            #Return translation using our trained model
            translations = decoding(model, x, batch, None, start_symbol=1, max_len=args.decoding_length, method=args.decoding, n_beam=args.n_beam, device=device)

            #Loop over translations and references
            #NOTE: discard special tokens
            for j in range(y.shape[0]):
                ys = y[j, :]
                ys = ys[ys != 0]
                #Keep <eos> token
                ys = ys[1:]

                translation = translations[j]

                #Convert index tokens to words
                translation_corpus.append([vocab[x.item()] for x in translation])
                references_corpus.append([[vocab[x.item()] for x in ys]])
                reference_corpus.append([vocab[x.item()] for x in ys])

        #Free some memory
        #NOTE: this helps alot in avoiding cuda out of memory
        del x, y, batch

    assert len(translation_corpus) == len(references_corpus)

    if(args.save):
        #Save results in txt files
        str1=" "
        with open(os.path.join(save_path, 'translations.txt') ,'a') as trans_file:
            trans_file.write(str1.join(translation_corpus)+'\n')

        str1=" "
        with open(os.path.join(save_path, 'references.txt'), 'a') as ref_file:
            ref_file.write(str1.join(reference_corpus)+'\n')

    if(args.bleu):

        #Default return
        #NOTE: you can use the ntlk library to measure the bleu score
        #bleu_4 = corpus_bleu(reference_corpus, translation_corpus)
        bleu_4, _, _, _, _, _ = compute_bleu(references_corpus, translation_corpus, max_order=4)

        #weights = (1.0/1.0, )
        bleu_1, _, _, _, _, _ = compute_bleu(references_corpus, translation_corpus, max_order=1)

        #weights = (1.0/2.0, 1.0/2.0, )
        #bleu_2 = corpus_bleu(reference_corpus, translation_corpus, weights)
        bleu_2, _, _, _, _, _ = compute_bleu(references_corpus, translation_corpus, max_order=2)

        #weights = (1.0/3.0, 1.0/3.0, 1.0/3.0,)
        #bleu_3 = corpus_bleu(reference_corpus, translation_corpus, weights)
        bleu_3, _, _, _, _, _ = compute_bleu(references_corpus, translation_corpus, max_order=3)

        log_str = 'Bleu Evaluation: ' + '\t' \
        + 'Bleu_1: ' + str(bleu_1) + '\t' \
        + 'Bleu_2: ' + str(bleu_2) + '\t' \
        + 'Bleu_3: ' + str(bleu_3) + '\t' \
        + 'Bleu_4: ' + str(bleu_4)

        print(log_str)

        if(args.save):
            #Save evaluation results in a log file
            with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
                f.write(log_str+'\n')

    if(args.rouge):

        reference_corpus = [" ".join(reference) for reference in reference_corpus]
        translation_corpus = [" ".join(hypothesis) for hypothesis in translation_corpus]

        score = rouge(translation_corpus, reference_corpus)
        print(score["rouge_l/f_score"])

        log_str = 'Rouge Evaluation: ' + '\t'
        print(log_str)

        if(args.save):
            #Save evaluation results in a log file
            with open(os.path.join(args.save_path, 'log.txt'), 'a') as f:
                f.write(log_str+'\n')

