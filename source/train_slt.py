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
import csv

from torch.optim.lr_scheduler import StepLR, MultiStepLR

from transformer_slt import make_model as TRANSFORMER
from dataloader_slt import loader
from utils import path_data, Batch, LabelSmoothing, greedy_decode, NoamOpt

#Progress bar to visualize training progress
import progressbar

import matplotlib.pyplot as plt

#For model summary
from torchsummary import summary

#Plotting
from viz import learning_curve_slt

#Visualize GPU resources
import GPUtil

#Produce translation for blue evaluation
import bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

#For Word Error Rate
from jiwer import wer

#NOTE: use tf CTC decoder for decoding
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Training the transformer-like network')

parser.add_argument('--data', type=str, default=os.path.join('data','PHOENIX-2014-T'),
                    help='location of the data corpus')

parser.add_argument('--data_type', type=str, default='keyfeatures',
                    help='features/resized_features/keyfeatures.')

parser.add_argument('--fixed_padding', type=int, default=None,
                    help='None/64')

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slt_lookup.txt'),
                    help='location of the words lookup table')

#If we want to train using gloss annotation + translation annotation
parser.add_argument('--lookup_table_gloss', type=str, default=os.path.join('data','slt_lookup_gloss.txt'),
                    help='location of the word gloss lookup table')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images.')

##For data augmentation
parser.add_argument('--augmentation', action='store_true',
                    help='Apply image augmentation.')

#Put to 0 to avoid memory segementation fault
parser.add_argument('--num_workers', type=int, default=4,
                    help='NOTE: put num of workers to 0 to avoid memory saturation.')

parser.add_argument('--show_sample', action='store_true',
                    help='Show a sample a preprocessed data (sequence of image of sign + translation).')

parser.add_argument('--optimizer', type=str, default='ADAM',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM / NOAM')

parser.add_argument('--scheduler', type=str, default=None,
                    help='Type of scheduler, multi-step or stepLR')

parser.add_argument('--milestones', default="10,30", type=str,
                    help="milestones for MultiStepLR or stepLR")

parser.add_argument('--batch_size', type=int, default=2,
                    help='size of one minibatch')

parser.add_argument('--initial_lr', type=float, default=0.00001,
                    help='initial learning rate')

parser.add_argument('--hidden_size', type=int, default=1280,
                    help='size of hidden layers. NOTE: This must be a multiple of n_heads.')

parser.add_argument('--save_best', action='store_true',
                    help='save the model w/ the best validation performance')

parser.add_argument('--num_layers', type=int, default=4,
                    help='number of transformer blocks')

parser.add_argument('--n_heads', type=int, default=10,
                    help='number of self attention heads')

parser.add_argument('--pretrained', type=bool, default=True,
                    help='embedding layers are pretrained using imagenet')

parser.add_argument('--hand_query', action='store_true',
                    help='Set hand as a query for transformer network.')

parser.add_argument('--emb_type', type=str, default='2d',
                    help='Type of image embeddings 2d or 3d.')

parser.add_argument('--emb_network', type=str, default='mb2',
                    help='Image embeddings network: mb2/resnet')

parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs to stop after')

parser.add_argument('--dp_keep_prob', type=float, default=0.7,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--valid_steps', type=int, default=1, help='Do validation each valid_steps time!')

parser.add_argument('--save_steps', type=int, default=10, help='Save model after each N epoch')

parser.add_argument('--debug', action='store_true')

parser.add_argument('--save_dir', type=str, default='EXPERIMENTATIONS',
                    help='path to save the experimental config, logs, model')

parser.add_argument('--evaluate', action='store_true',
                    help="Evaluate dev set using bleu metric each epoch.")

parser.add_argument('--resume', action='store_true',
                    help="Resume training from a checkpoint.")

parser.add_argument('--checkpoint',type=str, default='',
                    help="resume training from a previous checkpoint.")

parser.add_argument('--label_smoothing', type=float, default=None,
                    help="label smoothing loss.")

parser.add_argument('--hybrid', action='store_true',
                    help="Train using gloss and translation annotation.")

parser.add_argument('--rel_window', type=int, default=None,
                    help="Use local masking window.")

#----------------------------------------------------------------------------------------


## SET EXPERIMENTATION AND SAVE CONFIGURATION

#Same seed for reproducibility)
parser.add_argument('--seed', type=int, default=1111, help='random seed')

#Save folder with the date
start_date = dt.datetime.now().strftime("%Y-%m-%d-%H.%M")
print ("Start Time: "+start_date)

args = parser.parse_args()

#Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#experiment_path = PureWindowsPath('EXPERIMENTATIONS\\' + start_date)
experiment_path = os.path.join('EXPERIMENTATIONS',start_date)

# Creates an experimental directory and dumps all the args to a text file
if(os.path.exists(experiment_path)):
    print('Experiment already exists..')
    quit(0)
else:
    os.makedirs(experiment_path)

print ("\nPutting log in EXPERIMENTATIONS/%s"%start_date)

args.save_dir = os.path.join(args.save_dir, start_date)

#Dump all configurations/hyperparameters in txt
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    f.write('Experimentation done at: '+ str(start_date)+' with current configurations:\n')
    for arg in vars(args):
        f.write(arg+' : '+str(getattr(args, arg))+'\n')

#-------------------------------------------------------------------------------
#Run on GPU
if torch.cuda.is_available():
    print("Training on GPU!")
    device = torch.device("cuda:0")
else:
#Run on CPU
    print("WARNING: Training on CPU, this will likely run out of memory, Go buy yourself a GPU!")
    device = torch.device("cpu")
#--------------------------------------------------------------------------------



#Computation for one epoch
def run_epoch(model, data, is_train=False, device='cuda:0', n_devices=1):

    if is_train:
        model.train()  # Set model to training mode
        print ("Training..")
        phase='train'
    else:
        model.eval()   # Set model to evaluate mode
        print ("Evaluating..")
        phase='valid'

    start_time = time.time()

    loss = 0.0
    total_loss = 0.0
    total_tokens = 0
    total_seqs = 0
    tokens = 0
    total_correct = 0.0
    n_correct = 0.0

    total_wer_score = 0.0
    sentence_count = 0

    targets = []
    hypotheses = []

    #For progress bar
    bar = progressbar.ProgressBar(maxval=dataset_sizes[phase], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    j = 0

    #Loop over minibatches
    for step, (x, x_lengths, y, y_lengths, gloss, gloss_lengths) in enumerate(data):

        #Update progress bar with every iter
        j += len(x)
        bar.update(j)

        if(type(gloss) != type(None)):
            gloss = torch.from_numpy(gloss).to(device)

        y = torch.from_numpy(y).to(device)
        x = x.to(device)

        #NOTE: clone y to avoid overridding it
        batch = Batch(x_lengths, y_lengths, None, y.clone(), emb_type=args.emb_type, DEVICE=device, fixed_padding=args.fixed_padding, rel_window=args.rel_window)

        model.zero_grad()

        #Return tuple of (output, encoder_output)
        #output = (batch_size, tgt_seq_length, tgt_vocab_size)
        #encoder_output = (batch_size, input_seq_length, trg_vocab_size)
        if(args.hybrid):
            output, encoder_output = model.forward(x, batch.trg, batch.src_mask, batch.trg_mask, batch.rel_mask, None)

            #CTC loss expects (batch, trg_seq, trg_vocab)
            encoder_output = encoder_output.transpose(0,1)
        else:
            output = model.forward(x, batch.trg, batch.src_mask, batch.trg_mask, batch.rel_mask, None)

        #Produce translation for blue score
        #Evaluate on dev
        if(is_train==False):

            x = Variable(x)

            translations = greedy_decode(model, x, None, batch.rel_mask, batch.src_mask,
                            max_len=20, start_symbol=1, device=device)

            #Loop over batch to create sentences
            for i in range(len(y)):

                ys = y[i, :]
                ys = ys[ys != 0]
                #NOTE: keep eos
                ys = ys[1:]

                translation = translations[i]

                hyp_trans = [vocab[x.item()] for x in translation]
                gt_trans = [vocab[x.item()] for x in ys]

                translation_corpus.append(hyp_trans)
                #NOTE: required to list of list (since we have 1 reference for each gt sentence)
                reference_corpus.append([gt_trans])

        x_lengths = torch.IntTensor(x_lengths)
        y_lengths = torch.IntTensor(y_lengths)

        if(type(gloss_lengths) != type(None)):
            gloss_lengths = torch.IntTensor(gloss_lengths)

        #Get CTCloss of batch without averaging
        if(args.hybrid):
            loss_ctc = ctc_loss(encoder_output, gloss.cpu(), x_lengths.cpu(), gloss_lengths.cpu())

        #Remove sos tokens from y
        y = y[:, 1:]

        #Predicted words with highest prob
        _, pred = torch.max(output, dim=-1)

        #NOTE: dont count pad
        for i in range(y.shape[0]):
            n_correct += (pred[i, :y_lengths[i]-1] ==  y[i, :y_lengths[i]-1]).sum()

        #NOTE: The transformer is an auto-regressive model: it makes predictions one part at a time,
        #and uses its output so far to decide what to do next
        #Teacher forcing is passing the true output to the next time step regardless of what the model predicts at the current time step.

        #Input of decoder (with sos and without eos)
        #Target (without sos and with eos)

        #NOTE: pred must be same shape as y
        y = y.contiguous().view(-1)
        pred = pred.contiguous().view(-1)
        output = output.view(-1, vocab_size)

        assert y.shape == pred.shape

        #Get loss cross entropy (from decoder) of batch without averaging
        loss = loss_fn(output, y)

        if(args.hybrid):
            #Joint CTC/Decoder loss
            loss = loss + loss_ctc

        total_loss += loss
        total_seqs += batch.seq
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        total_correct += n_correct

        if is_train:

            loss.backward()

            #Weight clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if step % 100 == 0:
                elapsed = time.time() - start_time
                print("Step: %d, Loss: %f, Frame per Sec: %f, Token per sec: %f, Word Accuracy: %f" %
                      (step, loss / batch.ntokens, total_seqs * batch_size / elapsed, tokens / elapsed, n_correct.item() / tokens.item()))

                start_time = time.time()
                total_seqs = 0
                tokens = 0
                n_correct = 0.0

        #Free some memory
        #NOTE: this helps alot in avoiding cuda out of memory
        del loss, output, y

    if(is_train):
        print("Total word Accuracy: %f" %
                    (total_correct.item() / total_tokens.item()))
        return total_loss.item() / total_tokens.item()
    else:
        return translation_corpus, reference_corpus, total_loss.item() / total_tokens.item(), total_correct.item() / total_tokens.item()

#-------------------------------------------------------------------------------------------------------

### LOAD DATALOADERS

# In debug mode, try batch size of 1
if args.debug:
    batch_size = 1
else:
    batch_size = args.batch_size

#TO DO: add as hyperparameter
train_path, valid_path, test_path = path_data(data_path=args.data, task='SLT', features_type=args.data_type, hand_query=args.hand_query)

#Pass the annotation + image sequences locations
train_dataloader, train_size = loader(csv_file=train_path[1],
                root_dir=train_path[0],
                lookup=args.lookup_table,
                lookup_gloss=args.lookup_table_gloss,
                rescale = args.rescale,
                augmentation = args.augmentation,
                batch_size = batch_size,
                num_workers = args.num_workers,
                show_sample = args.show_sample,
                istrain=True,
                fixed_padding=args.fixed_padding,
                hand_dir=train_path[2]
                )

#No data augmentation for valid data
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
                fixed_padding=args.fixed_padding,
                hand_dir=valid_path[2]
                )

print('Dataset sizes:')
dataset_sizes = {}
dataset_sizes.update({'train':train_size})
dataset_sizes.update({'valid':valid_size})
print(dataset_sizes)


if(args.lookup_table_gloss and args.hybrid):
    #Retrieve size of target vocab
    with open(args.lookup_table_gloss, 'rb') as pickle_file:
        vocab_gloss = pickle.load(pickle_file)

    gloss_vocab_size = len(vocab_gloss)

    #Switch keys and values of vocab to easily look for words
    vocab_gloss = {y:x for x,y in vocab_gloss.items()}

    #You should find 1084 + 2 (pad + unk)
    print('Vocabulary Glosses size:' + str(gloss_vocab_size))
else:
    gloss_vocab_size = None
    vocab_gloss = None

#Retrieve size of target vocab
with open(args.lookup_table, 'rb') as pickle_file:
   vocab = pickle.load(pickle_file)

vocab_size = len(vocab)

#Switch keys and values of vocab to easily look for words
vocab = {y:x for x,y in vocab.items()}

#You should find 2887 + 4 (pad + unk + eos + sos)
print('Vocabulary Translation size:' + str(vocab_size))

#-----------------------------------------------------------------------------------------------------------------

#Load the whole model
model = TRANSFORMER(tgt_vocab=vocab_size, gloss_vocab=gloss_vocab_size, n_stacks=args.num_layers, n_units=args.hidden_size,
                            n_heads=args.n_heads, d_ff=2048, dropout=1.-args.dp_keep_prob, image_size=args.rescale, pretrained=args.pretrained, emb_type=args.emb_type, emb_network=args.emb_network)


#Load model on GPU or multiple GPUs
if torch.cuda.device_count() > 1:
    #How many GPUs you are using
    n_devices = torch.cuda.device_count()
    print("Using ", n_devices, "GPUs!, Let's GO!")
    #Making your model run parallelly on multiple GPUs
    #NOTE: Pytorch divides data to even chunks across GPUs
    model = nn.DataParallel(model)

else:
    print("Training using 1 device (GPU/CPU), use very small batch_size!")
    #Load model into device (GPU OR CPU)
    n_devices = 1


model = model.to(device)
print("Loading to GPUs")
print(GPUtil.showUtilization())


if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
elif args.optimizer == 'noam':
    optimizer = NoamOpt(args.hidden_size, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

if args.scheduler == 'multi-step':
    milestones = [int(v.strip()) for v in args.milestones.split(",")]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

elif args.scheduler == 'stepLR':
    scheduler = StepLR(optimizer, step_size=args.milestones, gamma=0.1)
else:
    print('No scheduler!')


train_ppls = []
train_losses = []
val_ppls = []
val_losses = []
ns_words = []
bleu_1s = []
bleu_2s = []
bleu_3s = []
bleu_4s = []

best_val_so_far = np.inf
best_bleu = 0.0
best_acc_so_far = 0.0
times = []

# In debug mode, only run one epoch
if args.debug:
    num_epochs = 1
else:
    num_epochs = args.num_epochs

#Load weights from previous training session
if(args.resume):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    loss_fn = checkpoint['loss']
    best_bleu = checkpoint['best_bleu']

else:
    start_epoch = 0
    if(args.label_smoothing):
        loss_fn = LabelSmoothing(size=len(vocab), padding_idx=0, smoothing=args.label_smoothing)
    else:
        loss_fn = nn.NLLLoss(ignore_index=0, reduction='sum')

if(vocab_gloss):
    #Use blank label as padding
    #NOTE: blank label = Nclasses - 1
    blank_index = len(vocab_gloss) - 1
    ctc_loss = nn.CTCLoss(blank=blank_index, reduction='sum', zero_infinity=True)

###
#Main Training loop

for epoch in range(start_epoch, num_epochs):

    start = time.time()

    print('\nEPOCH '+str(epoch)+' ------------------')

    # RUN MODEL ON TRAINING DATA
    train_loss = run_epoch(model, train_dataloader, True, device=device)
    print("After train epoch..")
    print(GPUtil.showUtilization())

    #Save perplexity
    train_ppl = np.exp(train_loss)

    if(args.scheduler):
        scheduler.step()

    if(epoch % args.valid_steps == 0):

        #Use it for evaluation with blue
        translation_corpus = []
        reference_corpus = []

        print("Evaluate Blue..")

        # RUN MODEL ON VALIDATION DATA
        with torch.no_grad():
            translation_corpus, reference_corpus, val_loss, word_acc = run_epoch(model, valid_dataloader, device=device, n_devices=1)

        val_ppl = np.exp(val_loss)

        #Default return bleu-4
        bleu_4 = corpus_bleu(reference_corpus, translation_corpus)

        weights = (1.0/1.0, )
        bleu_1 = corpus_bleu(reference_corpus, translation_corpus, weights)

        weights = (1.0/2.0, 1.0/2.0,)
        bleu_2 = corpus_bleu(reference_corpus, translation_corpus, weights)

        weights = (1.0/3.0, 1.0/3.0, 1.0/3.0,)
        bleu_3 = corpus_bleu(reference_corpus, translation_corpus, weights)

        log_str = 'epoch: ' + str(epoch) + '\t' \
            + 'Bleu-1: ' + str(bleu_1) + '\t' \
            + 'Bleu-2: ' + str(bleu_2) + '\t' \
            + 'Bleu-3: ' + str(bleu_3) + '\t' \
            + 'Bleu-4: ' + str(bleu_4)

        print(log_str)

        if best_bleu < bleu_4:
            best_bleu = bleu_4

            #if args.save_best:
            print("Saving entire model with best params")
            torch.save(model, os.path.join(args.save_dir, 'best_params.pt'))

        if word_acc > best_acc_so_far:
            best_acc_so_far = word_acc

        if(epoch % args.save_steps == 0):
            #Save after each epoch and save optimizer state
            print("Saving model parameters for epoch: "+str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'best_acc': best_acc_so_far,
                'best_bleu': best_bleu
                },
                os.path.join(args.save_dir, 'epoch_'+str(epoch)+'_bleu_'+str(bleu_4)+'.pt'))

        # SAVE RESULTS
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        times.append(time.time() - start)
        ns_words.append(word_acc)
        bleu_1s.append(bleu_1)
        bleu_2s.append(bleu_2)
        bleu_3s.append(bleu_3)
        bleu_4s.append(bleu_4)


        log_str = 'epoch: ' + str(epoch) + '\t' \
             + 'train ppl: ' + str(train_ppl) + '\t' \
             + 'val ppl: ' + str(val_ppl) + '\t' \
             + 'train loss: ' + str(train_loss) + '\t' \
             + 'val loss: ' + str(val_loss) + '\t' \
             + 'word acc: ' + str(word_acc) + '\t' \
             + 'best word acc: ' + str(best_acc_so_far) + '\t' \
            + 'bleu-1 score: ' + str(bleu_1) + '\t' \
            + 'bleu-2 score: ' + str(bleu_2) + '\t' \
            + 'bleu-3 score: ' + str(bleu_3) + '\t' \
            + 'bleu-4 score: ' + str(bleu_4) + '\t' \
            + 'best bleu score: ' + str(best_bleu) + '\t' \
            + 'time (s) spent in epoch: ' + str(times[-1])

        print(log_str)
        with open (os.path.join(args.save_dir, 'log.txt'), 'a') as f_:
            f_.write(log_str+ '\n')

        #SAVE LEARNING CURVES
        lc_path = os.path.join(args.save_dir, 'learning_curves.npy')
        print('\nDONE\n\nSaving learning curves to '+lc_path)
        np.save(lc_path, {'train_ppls':train_ppls,
                  'val_ppls':val_ppls,
                  'train_losses':train_losses,
                   'val_losses':val_losses,
                   'word_acc':ns_words,
                  'bleu_1':bleu_1s,
                  'bleu_2':bleu_2s,
                  'bleu_3':bleu_3s,
                  'bleu_4':bleu_4s
                  })


        print("Saving plots")
        learning_curve_slt(args.save_dir)

        #Reach convergence
        if(train_ppl == 1):
            print('Yay!')
            break

