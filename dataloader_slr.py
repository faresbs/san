
#############################################
#                                           #
# Load sequential data from PHOENIX-2014    #
#                                           #
#############################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import io, transform

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import scipy.misc

#Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def collate_fn(data, fixed_padding=None, pad_index=1232):
    """Creates mini-batch tensors w/ same length sequences by performing padding to the sequecenses.
    We should build a custom collate_fn to merge sequences w/ padding (not supported in default).
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding), else pad
    all Sequences to a fixed length.

    Returns:
        hand_seqs: torch tensor of shape (batch_size, padded_length).
        hand_lengths: list of length (batch_size); 
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); 
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); 
    """

    def pad(sequences, t):
        lengths = [len(seq) for seq in sequences]

        #For sequence of images
        if(t=='source'):
            #Retrieve shape of single sequence
            #(seq_length, channels, n_h, n_w)
            seq_shape = sequences[0].shape
            if(fixed_padding):
                padded_seqs = fixed_padding
                padded_seqs = torch.zeros(len(sequences), fixed_padding, seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])
            else:
                padded_seqs = torch.zeros(len(sequences), max(lengths), seq_shape[1], seq_shape[2], seq_shape[3]).type_as(sequences[0])

        #For sequence of words
        elif(t=='target'):
            padded_seqs = np.full((len(sequences), max(lengths)), fill_value=pad_index, dtype=np.int)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]

        return padded_seqs, lengths

    src_seqs = []
    trg_seqs = []
    right_hands = []
    left_hands = []

    for element in data:
        src_seqs.append(element['images'])
        trg_seqs.append(element['translation'])

        right_hands.append(element['right_hands'])

    #pad sequences
    src_seqs, src_lengths = pad(src_seqs, 'source')
    trg_seqs, trg_lengths = pad(trg_seqs, 'target')

    #pad hand sequences
    if(type(right_hands[0]) != type(None)):
        hand_seqs, hand_lengths = pad(right_hands, 'source')
    else:
        hand_seqs = None
        hand_lengths = None

    return src_seqs, src_lengths, trg_seqs, trg_lengths, hand_seqs, hand_lengths


#From abstract function Dataset
class PhoenixDataset(Dataset):
    """Sequential Sign language images dataset."""

    def __init__(self, csv_file, root_dir, lookup_table, transform=None, rescale=224, sos_index=1, eos_index=2, unk_index=0, fixed_padding=None, hand_dir=None, hand_transform=None):

        #Get data
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.hand_dir = hand_dir

        self.transform = transform
        self.hand_transform = hand_transform

        self.rescale = rescale

        #index used for eos token and unk
        self.eos_index = eos_index
        self.unk_index = unk_index
        self.sos_index = sos_index

        #Retrieve lookup table dic from path
        with open(lookup_table, 'rb') as pickle_file:
            self.lookup_table = pickle.load(pickle_file)


    def __len__(self):
        #Return size of dataset
        return len(self.annotations)

    def __getitem__(self, idx):

        #Retrieve the name id of sequence from csv annotations
        name = self.annotations.iloc[idx, 0].split('|')[0]

        seq_name = os.path.join(self.root_dir, name)

        for path, d, files in os.walk(seq_name):

            seq_length = len(files)

            trsf_images = torch.zeros((seq_length, 3, self.rescale, self.rescale))

            #Get hand cropped image list if exists
            if(self.hand_dir):
                hand_path = os.path.join(self.hand_dir, name)
                hand_images = torch.zeros((seq_length, 3, 112, 112))
            else:
                hand_images = None

            #Save the images of seq
            for i in range(1, seq_length):
                img_name = os.path.join(path, 'images'+'{:04d}'.format(i)+'.png')
                #print(img_name)
                if(self.hand_dir):
                    hand_name_0 = os.path.join(hand_path, 'images'+'{:04d}'.format(i)+'.png')
                    #print(hand_name_0)
                    if(io.imread(hand_name_0).shape[2] == 3):
                        hand_images[i-1] = self.hand_transform(io.imread(hand_name_0))
                    else:
                        hand_images[i-1] = self.hand_transform(io.imread(hand_name_0)[:, :, :3])

                #print(Image.open(img_name))
                #print("HERE")
                #NOTE: some images got shape of (260, 220, 4)
                if(io.imread(img_name).shape[2] == 3):
                    trsf_images[i-1] = self.transform(io.imread(img_name, plugin='pil'))
                else:
                    trsf_images[i-1] = self.transform(io.imread(img_name, plugin='pil')[:, :, :3])

        #Retrive the translation (ground truth text translation) from csv annotations
        translation = self.annotations.iloc[idx, 0].split('|')[-1]

        #Split translation phrase to set of words
        translation = translation.split(' ')

        #Save index values of the words
        trans = []

        #Add current words in lookup table
        for word in translation:
            #Get index of the current word if it exists in dict
            if(word in self.lookup_table.keys()):
                trans.append(self.lookup_table[word])
            else:
                #If words doesnt exist in train vocab then <unk>
                trans.append(self.unk_index)

        #NOTE: full frame seq and hand seq should be with the same seq length
        sample = {'images': trsf_images, 'right_hands':hand_images, 'translation': trans}

        return sample


# Helper function to show a batch
def show_batch(sample_batched):
    """Show sequence of images with translation for a batch of samples."""

    images_batch, images_length, trans_batch, trans_length = \
            sample_batched
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    #Show only one sequence of the batch
    grid = utils.make_grid(images_batch[0, :images_length[0]])
    grid = grid.numpy()
    return np.transpose(grid, (1,2,0))


#Use this to subtract mean from each pixel measured from PHOENIX-T dataset
#Note: means has been subtracted from 227x227 images, this has been provided by camgoz
class SubtractMeans(object):
    def __init__(self, path, rescale):
        #NOTE: Newest np versions default value allow_pickle=False
        self.mean = np.load(path, allow_pickle=True)
        self.mean = self.mean.astype('uint8')
        self.rescale = rescale

    def __call__(self, image):

        #No need to resize (take long time..)
        #image = cv2.resize(image,(self.mean.shape[0], self.mean.shape[1]))
        assert image.shape == self.mean.shape
        image -= self.mean
        #image = cv2.resize(image,(self.rescale, self.rescale))

        return image


def loader(csv_file, root_dir, lookup, rescale, augmentation, batch_size, num_workers, show_sample, istrain=False, mean_path='FulFrame_Mean_Image_227x227.npy', fixed_padding=None, hand_dir=None):

    #Note: when using random cropping, this with reshape images with randomCrop size instead of rescale
    if(augmentation and istrain):

        trans = transforms.Compose([
            SubtractMeans(mean_path, rescale),
            transforms.ToPILImage(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
            transforms.Resize((rescale, rescale)),
            transforms.ToTensor()])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        hand_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(10),
                transforms.Resize((112, 112)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    else:
        trans = transforms.Compose([
            SubtractMeans(mean_path, rescale),
            transforms.ToPILImage(),
            transforms.Resize((rescale, rescale)),
            transforms.ToTensor()])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        hand_trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    ##Iterate through the dataset and apply data transformation on the fly

    #Apply data augmentation to avoid overfitting
    transformed_dataset = PhoenixDataset(csv_file=csv_file,
                                            root_dir=root_dir,
                                            lookup_table=lookup,
                                            transform=trans,
                                            rescale=rescale,
                                            hand_dir=hand_dir,
                                            hand_transform=hand_trans
                                            )

    size = len(transformed_dataset)

    #Iterate in batches
    #Note: put num of workers to 0 to avoid memory saturation
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    #Show a sample of the dataset
    if(show_sample and istrain):
        for i_batch, sample_batched in enumerate(dataloader):
            #plt.figure()
            img = show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.imshow(img)
            #plt.show()
            plt.savefig('data_sample.png')
            break

    return dataloader, size
