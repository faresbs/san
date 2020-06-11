import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import datetime as dt
import _pickle as pickle
import csv

from torch.optim.lr_scheduler import StepLR, MultiStepLR

from transformer_slr import make_model as TRANSFORMER
from dataloader_slr import loader
from utils import path_data, Batch, LabelSmoothing, NoamOpt

#Progress bar to visualize training progress
import progressbar

import matplotlib.pyplot as plt

#For model summary
from torchsummary import summary

#Plotting
from viz import learning_curve_slr

#Visualize GPU resources
import GPUtil

#Lavenghtein distance (WER)
from jiwer import wer

#NOTE: use tf CTC decoder for decoding
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

###
# Arg parsing
##############

parser = argparse.ArgumentParser(description='Training the transformer-like network')

parser.add_argument('--data', type=str, default=os.path.join('data','phoenix-2014.v3','phoenix2014-release','phoenix-2014-multisigner'),
                   help='location of the data corpus')

parser.add_argument('--data_type', type=str, default='keyfeatures',
                    help='features/resized_features/keyfeatures.')

parser.add_argument('--fixed_padding', type=int, default=None,
                    help='None/64')

parser.add_argument('--lookup_table', type=str, default=os.path.join('data','slr_lookup.txt'),
                    help='location of the words lookup table')

parser.add_argument('--rescale', type=int, default=224,
                    help='rescale data images.')
##For data augmentation
parser.add_argument('--augmentation', action="store_true",
                    help='Apply image augmentation')

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

parser.add_argument('--initial_lr', type=float, default=0.0001,
                    help='initial learning rate')

parser.add_argument('--hidden_size', type=int, default=1280,
                    help='size of hidden layers. NOTE: This must be a multiple of n_heads.')

parser.add_argument('--save_best', action='store_true',
                    help='save the model w/ the best validation performance')

parser.add_argument('--num_layers', type=int, default=2,
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
                    help='Image embeddings network: mb2/i3d/m3d')

parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs to stop after')

parser.add_argument('--dp_keep_prob', type=float, default=0.7,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

parser.add_argument('--valid_steps', type=int, default=1, help='Do validation each valid_step')

parser.add_argument('--save_steps', type=int, default=10, help='Save model after each N epoch')

parser.add_argument('--debug', action='store_true')

parser.add_argument('--save_dir', type=str, default='EXPERIMENTATIONS',
                    help='path to save the experimental config, logs, model')

parser.add_argument('--evaluate', action='store_true',
                    help="Evaluate dev set using bleu metric each epoch.")

parser.add_argument('--resume', action='store_true',
                    help="Resume training from a checkpoint.")

parser.add_argument('--checkpoint',type=str, default=None,
                    help="resume training from a previous checkpoint.")

parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help="label smoothing loss.")

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
    batch_tokens = 0.0
    total_seqs = 0
    tokens = 0
    total_correct = 0.0
    n_correct = 0.0

    wer_score = 0.0
    total_wer_score = 0.0
    count = 0

    gt = []
    hyp = []

    #For progress bar
    bar = progressbar.ProgressBar(maxval=dataset_sizes[phase], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    j = 0

    #Loop over minibatches
    for step, (x, x_lengths, y, y_lengths, hand_regions, hand_lengths) in enumerate(data):

        #Update progress bar with every iter
        j += len(x)
        bar.update(j)

        y = torch.from_numpy(y).to(device)
        x = x.to(device)

        if(args.hand_query):
             hand_regions = hand_regions.to(device)
        else:
             hand_regions = None

        #NOTE: clone y to avoid overridding it
        batch = Batch(x_lengths, y_lengths, hand_lengths, trg=None, emb_type=args.emb_type, DEVICE=device, fixed_padding=args.fixed_padding, rel_window=args.rel_window)

        model.zero_grad()

        #Shape(batch_size, tgt_seq_length, tgt_vocab_size)
        #NOTE: no need for trg if we dont have a decoder
        output, output_context, output_hand = model.forward(x, batch.src_mask, batch.rel_mask, hand_regions)

        #CTC loss expects (Seq, batch, vocab)
        if(args.hand_query):
            output = output.transpose(0,1)
            output_context = output_context.transpose(0,1)
            output_hand = output_hand.transpose(0,1)
        else:
            output = output_context.transpose(0,1)

        x_lengths = torch.IntTensor(x_lengths)
        y_lengths = torch.IntTensor(y_lengths)


        if(is_train==False):

            #Run CTC beam decoder using tensorflow
            #NOTE: blank token in Tensorflow must be  (N-classes - 1)

            #Return tuple of sentences and probs
            decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=output.cpu().detach().numpy(),
                                 sequence_length=x_lengths.cpu().detach().numpy(), merge_repeated=False, beam_width=10, top_paths=1)
            #Get top 1 path
            #(batch, Seq)
            pred = decodes[0]

            #Transform sparse tensor to numpy
            pred = tf.sparse.to_dense(pred).numpy()

            for i in range(len(y)):

                #NOTE: we are doing log inside ctcdecoder
                #pred = (seq, beam, batch)

                ys = y[i, :y_lengths[i]]
                p = pred[i]

                hyp = (' '.join([vocab[x.item()] for x in p]))
                gt = (' '.join([vocab[x.item()] for x in ys]))

                total_wer_score += wer(gt, hyp, standardize=True)
                count += 1

        #output (Seq, batch, vocab_size)
        #y (batch, trg_size)
        #x_lengths (batch)
        #y_lengths (batch)

        #NOTE: produce Nan values if x length > y lengths
        #When extracting keyframes, make sure your src lengths are long enough or simply use zero infinity
        #Doing average loss here

        #IMPORTANT: Use Pytorch CTCloss
        loss = ctc_loss(output, y.cpu(), x_lengths.cpu(), y_lengths.cpu())

        if(args.hand_query):
            loss += ctc_loss(output_context, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            loss += ctc_loss(output_hand, y.cpu(), x_lengths.cpu(), y_lengths.cpu())
            loss = loss / 3

        total_loss += loss
        total_seqs += batch.seq
        total_tokens += (y != blank_index).data.sum()
        tokens += (y != blank_index).data.sum()
        batch_tokens += (y != blank_index).data.sum()

        if is_train:

            loss.backward()

            #Weight clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            if step % 100 == 0:
                elapsed = time.time() - start_time
                print("Step: %d, Loss: %f, Frame per Sec: %f, Token per sec: %f"%
                      (step, (loss / batch_tokens), total_seqs * batch_size / elapsed, tokens / elapsed))

                start_time = time.time()
                total_seqs = 0
                tokens = 0

        batch_tokens = 0.0

        #Free some memory
        #NOTE: this helps alot in avoiding cuda out of memory
        del loss, output, output_context, output_hand, y, hand_regions, batch

    if(is_train):
        print("Average Loss: %f" %(total_loss.item() / total_tokens.item()))
        return total_loss.item() / total_tokens.item()

    else:
        #Measure WER of all dataset
        print('Measuring WER..')
        print("Average WER: %f" %(total_wer_score/count))

        return total_loss.item() / total_tokens.item(), total_wer_score/count
#-------------------------------------------------------------------------------------------------------

### LOAD DATALOADERS

# In debug mode, try batch size of 1
if args.debug:
    batch_size = 1
else:
    batch_size = args.batch_size

train_path, valid_path, test_path = path_data(data_path=args.data, task='SLR', features_type=args.data_type, hand_query=args.hand_query)

#Pass the annotation + image sequences locations
train_dataloader, train_size = loader(csv_file=train_path[1],
                root_dir=train_path[0],
                lookup=args.lookup_table,
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
                rescale = args.rescale,
                augmentation = args.augmentation,
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

#Retrieve size of target vocab
with open(args.lookup_table, 'rb') as pickle_file:
   vocab = pickle.load(pickle_file)

vocab_size = len(vocab)

#Switch keys and values of vocab to easily look for words
vocab = {y:x for x,y in vocab.items()}

#You should find
print('vocabulary size:' + str(vocab_size))

#-----------------------------------------------------------------------------------------------------------------

#Load the whole model
model = TRANSFORMER(tgt_vocab=vocab_size, n_stacks=args.num_layers, n_units=args.hidden_size,
                            n_heads=args.n_heads, d_ff=2048, dropout=1.-args.dp_keep_prob, image_size=args.rescale, pretrained=args.pretrained,
                            emb_type=args.emb_type, emb_network=args.emb_network)

#Load model on GPU or multiple GPUs
if torch.cuda.device_count() > 1:
    #How many GPUs you are using
    n_devices = torch.cuda.device_count()
    print("Using ", n_devices, "GPUs!, Let's GO!")
    model = nn.DataParallel(model)
else:
    print("Training using 1 device (GPU/CPU), use very small batch_size!")
    #Load model into device (GPU OR CPU)
    n_devices = 1


model = model.to(device)
print("Loading to GPUs")
print(GPUtil.showUtilization())


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
best_err_so_far = 999.9
times = []

if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

elif args.optimizer == 'noam':
    optimizer = NoamOpt(args.hidden_size, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


# In debug mode, only run one epoch
if args.debug:
    num_epochs = 1
else:
    num_epochs = args.num_epochs

#Load weights from previous training session
#Resume training or start from start w/ pretrained weights
if(args.checkpoint):
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    if(args.resume):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_fn = checkpoint['loss']
        best_bleu = checkpoint['best_wer']

if(args.checkpoint == None or args.resume == False):
    start_epoch = 0

    if args.scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    elif args.scheduler == 'stepLR':
        scheduler = StepLR(optimizer, step_size=args.milestones, gamma=0.1)
    else:
        print('No scheduler!')

    if(args.label_smoothing):
        loss_fn = LabelSmoothing(size=len(vocab), padding_idx=0, smoothing=args.label_smoothing)
    else:
        loss_fn = nn.NLLLoss(ignore_index=0, size_average=False)

blank_index = len(vocab)-1

#zero_infinity to avoid having numerical instabilities
#NOTE: N-class - 1 is for BLANK token if we are using tensorflow decoder
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

        #RUN MODEL ON VALIDATION DATA
        #NOTE: Helps with avoiding memory saturation
        with torch.no_grad():
            val_loss, word_err = run_epoch(model, valid_dataloader)

            if word_err < best_err_so_far:
                best_err_so_far = word_err

                #if args.save_best:
                print("Saving entire model with best params")
                torch.save(model, os.path.join(args.save_dir, 'best_params.pt'))

        val_ppl = np.exp(val_loss)

        # SAVE RESULTS
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        times.append(time.time() - start)
        ns_words.append(word_err)

        log_str = 'epoch: ' + str(epoch) + '\t' \
             + 'train ppl: ' + str(train_ppl) + '\t' \
             + 'val ppl: ' + str(val_ppl) + '\t' \
             + 'train loss: ' + str(train_loss) + '\t' \
             + 'val loss: ' + str(val_loss) + '\t' \
             + 'WER: ' + str(word_err) + '\t' \
            + 'BEST WER: ' + str(best_err_so_far) + '\t' \
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
                   'wer':ns_words,
                  })

        print("Saving plots")
        learning_curve_slr(args.save_dir)

        #Save every model every 10 epoch
        if(epoch % args.save_steps == 0):
            #Save after each epoch and save optimizer state
            print("Saving model parameters for epoch: "+str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,
                'best_wer': best_err_so_far
                },
                os.path.join(args.save_dir, 'epoch_'+str(epoch)+'_wer_'+str(word_err)+'.pt'))


        #We reached convergence
        if(train_ppl <= 1):
            print("YAy!!")
            break
