#######################################################
#This script is for evaluating for the task of SLT    #
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
import cv2

from transformer_slr import make_model as TRANSFORMER

from dataloader_slr import loader #For SLR
from utils import path_data, Batch, greedy_decode

#Progress bar to visualize training progress
import progressbar

import matplotlib.pyplot as plt

#Evaluation metrics
from bleu import compute_bleu
from rouge import rouge
from beam_search import beam_decode

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
#https://pypi.org/project/py-rouge/
#import rouge

#Lavenshtein distance (WER)
from jiwer import wer

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Evaluation')


parser.add_argument('--data', type=str, default=os.path.join('data','phoenix-2014.v3', 'phoenix2014-release','phoenix-2014-multisigner'),
                    help='location of the test data corpus')

parser.add_argument('--model_path', type=str, default=os.path.join("EXPERIMENTATIONS"),
                    help='location of the test data corpus')

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slr_lookup.txt'),
                    help='location of the words lookup table')

##For data augmentation
parser.add_argument('--augmentation', type=int, default=False,
                    help='Apply augmentation.')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images. NOTE: use same image size as the training or else you get worse results.')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=0,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data.')

parser.add_argument('--batch_size', type=int, default=1,
                    help='size of one minibatch')

parser.add_argument('--save', action='store_true',
                    help='save the results of the evaluation')

parser.add_argument('--hand_query', action='store_true',
                    help='Set hand cropped image as a query for transformer network.')

parser.add_argument('--det_model_path', type=str, default='models',
                    help='Detection model model path. Note: detection model and transformer network are trained seperatly.')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/mb2-ssd/rcnn')

parser.add_argument('--decoding', type=str, default='greedy',
                    help='Decoding method (greedy/beam).')

parser.add_argument('--n_beam', type=int, default=4,
                    help='Beam width when using bean search for decoding.')

parser.add_argument('--rel_window', type=int, default=None)

parser.add_argument('--bleu', action='store_true',
                    help='Use bleu for evaluation.')

parser.add_argument('--rouge', action='store_true',
                    help='Use rouge for evaluation.')

parser.add_argument('--txt', type=str, default=None,
                    help='Run evaluation from txt files.')


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


train_path, valid_path, test_path = path_data(data_path=args.data, task='SLR', hand_query=args.hand_query)


#No data augmentation for test data
test_dataloader, test_size = loader(csv_file=test_path[1],
                root_dir=test_path[0],
                lookup=args.lookup_table,
                rescale = args.rescale,
                augmentation = None,
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
                rescale = args.rescale,
                augmentation = None,
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


#Load the whole model
#model = TRANSFORMER(tgt_vocab=vocab_size, n_stacks=2, n_units=1280,
#                            n_heads=10, d_ff=2048, dropout=0.3, image_size=224,
#                                                        emb_type='2d', emb_network='mb2')
#model.load_state_dict(torch.load(args.model_path)['state_dict'])

#Load entire model w/ weights
model = torch.load(args.model_path, map_location=device)

model = model.to(device)
print("Model successfully loaded")

model.eval()   # Set model to evaluate mode
print ("Evaluating..")

start_time = time.time()

#Loop through test and val sets
dataloaders = [valid_dataloader, test_dataloader]
sizes = [valid_size, test_size]
dataset = ['valid', 'test']

#Blank token index
blank_index = 1232

#Run evaluation from txt files
if(args.txt):
    for d in range(len(dataset)):

        print(d)

        #Hypotheses file
        with open(os.path.join(args.txt,'simpl_translations_'+dataset[d]+'.txt')) as f:
            hyp = f.read().splitlines()

        #Reference file
        with open(os.path.join(args.txt,'simpl_references_'+dataset[d]+'.txt')) as f:
            ref = f.read().splitlines()

        assert len(hyp) == len(ref)

        total_wer_score = 0.0
        count = 0
        #Measuring WER
        for i in range(len(ref)):
            total_wer_score += wer(ref[i], hyp[i], standardize=True)
            count += 1

        print(total_wer_score/count)

    quit(0)


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

    total_wer_score = 0.0
    count = 0

    #Loop over minibatches
    for step, (x, x_lengths, y, y_lengths, hand_regions, hand_lengths) in enumerate(dataloader):

        #Update progress bar with every iter
        i += len(x)
        bar.update(i)

        if(args.hand_query):
             hand_regions = hand_regions.to(device)
        else:
             hand_regions = None

        y = torch.from_numpy(y).to(device)
        x = x.to(device)

        batch = Batch(x_lengths, y_lengths, hand_lengths, trg=None, DEVICE=device, emb_type=args.emb_type, fixed_padding=None, rel_window=args.rel_window)

        #with torch.no_grad():
        if(True):
            output, output_context, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions)

            #CTC loss expects (Seq, batch, vocab)
            if(args.hand_query):
                output = output.transpose(0,1)
                output_context = output_context.transpose(0,1)
                output_hand = output_hand.transpose(0,1)
            else:
                output = output_context.transpose(0,1)

             #Predicted words with highest prob
            _, pred = torch.max(output, dim=-1)

            #Remove <BLANK>
            #pred = pred[pred != blank_index]

            if(True):
                output[17, 0, pred[17].item()].backward(retain_graph=True)

                # pull the gradients out of the images feature map
                #They should have same shape
                gradients = model.get_activations_gradient()
                activations = model.get_activations().detach()

                #print(gradients.shape)
                #print(activations.shape)
                #sd

                # pool the gradients across the channels
                pooled_gradients = torch.mean(gradients, dim=[2, 3])
                #print(pooled_gradients.shape)

                # weight the channels by corresponding gradients
                for i in range(57):
                    for j in range(1280):
                        activations[i, j, :, :] = activations[i, j, :, :] * pooled_gradients[i, j]

                #print(activations.shape)

                # average the channels of the activations
                heatmap = torch.mean(activations, dim=1).squeeze()
                maxi = torch.max(heatmap)

                # relu on top of the heatmap
                heatmap = np.maximum(heatmap.cpu().numpy(), 0)

                # normalize the heatmap
                heatmap /= maxi.cpu().numpy()
                #print(heatmap)
                #print(heatmap.shape)

                for i in range(heatmap.shape[0]):
                    #Get image
                    img = cv2.imread(os.path.join(args.data, 'keyfeatures/fullFrame-210x260px/dev/10January_2011_Monday_tagesschau_default-7', 'images'+'{:04d}'.format(i+1)+'.png'))
                    img = cv2.resize(img, (args.rescale, args.rescale))
                    h = heatmap[i]
                    h = cv2.resize(h, (args.rescale, args.rescale))
                    h = np.uint8(255 * h)
                    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)

                    assert img.shape == h.shape

                    #h = h*0.4 + img
                    h = cv2.addWeighted(h, 0.5, img, 0.8, 0)
                    cv2.imwrite("samples/heatmap"+str(i)+".png", h)

            sd

            x_lengths = torch.IntTensor(x_lengths)
            y_lengths = torch.IntTensor(y_lengths)

            decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=output.cpu().detach().numpy(),
                            sequence_length=x_lengths.cpu().detach().numpy(), merge_repeated=False, beam_width=10, top_paths=1)

            pred = decodes[0]

            pred = tf.sparse.to_dense(pred).numpy()

            #Loop over translations and references

            for j in range(len(y)):

                ys = y[j, :y_lengths[j]]
                p = pred[j]

                #Remove <UNK> token
                p = p[p != 0]
                ys = ys[ys != 0]

                hyp = (' '.join([vocab[x.item()] for x in p]))
                gt = (' '.join([vocab[x.item()] for x in ys]))

                total_wer_score += wer(gt, hyp, standardize=True)
                count += 1

                #Convert index tokens to words
                translation_corpus.append(hyp)
                reference_corpus.append(gt)

        #Free some memory
        #NOTE: this helps alot in avoiding cuda out of memory
        del x, y, batch

    assert len(translation_corpus) == len(reference_corpus)

    print('WER score:'+str(total_wer_score/count))

    if(args.save):
        #Save results in txt files
        with open(os.path.join(save_path, 'translations_'+dataset[d]+'.txt') ,'w') as trans_file:
            trans_file.write("\n".join(translation_corpus))

        with open(os.path.join(save_path, 'references_'+dataset[d]+'.txt'), 'w') as ref_file:
            ref_file.write("\n".join(reference_corpus))

    if(args.bleu):

        #Default return
        #NOTE: bleu score of camgoz results is slightly better than ntlk -> use it instead
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

