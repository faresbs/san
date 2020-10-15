"""
Create lookup table of vocabulary
"""

import pandas as pd
import _pickle as pickle
import os

##Create lookup table for target word
#NOTE: We shouldnt take words of test and dev sets
#Dont take the . token word from the dataset
#We don't have SOS and EOS tokens for SLR

def lookup(path, save=True, train_file='train.corpus.csv', name='slt_lookup', only_train_words=True, remove_singletons=False, gloss=True):

    #Dict to save words with corresponding index
    table = {}
    table_gloss = {}
    counter = {}
    counter_gloss = {}


    #Loop over datasets(train, dev and test)
    for csv_file in os.listdir(path):

        #Take only train annotations
        if(only_train_words):
            if csv_file != train_file:
                continue

        #Load annotations data from csv
        annotations = pd.read_csv(os.path.join(path, csv_file))

        #Loop over the annotations examples
        for i in range(len(annotations)):

            #Retrive the translation (ground truth text translation) from current annotation
            translation = annotations.iloc[i, 0].split('|')[-1]

            #Retrieve the gloss annotation
            #NOTE: use this if we want to use both translation and glosses for training
            if(gloss):
                glosses = annotations.iloc[i, 0].split('|')[-2]
                glosses = glosses.split(' ')

                #UNK token for words not seen in training
                if(only_train_words):
                    table_gloss.update({'<UNK>':0})

                #Loop over the words in current glosses
                for word in glosses:
                    #Check if word doesnt already exists in dict
                    if((word in table_gloss.keys()) == False):
                        #Save word w/ index (current length of table)
                        #Note: start from index 1, first indexes unk
                        table_gloss.update({word:len(table_gloss)})
                        counter_gloss.update({word:1})
                    else:
                        counter_gloss[word] += 1

            #Split translation phrase to set of words
            translation = translation.split(' ')

            #Set pad as first index 0
            table.update({'<PAD>':0})

            #Start-of-sentence and end-of-sentence tokens
            table.update({'<SOS>':1})
            table.update({'<EOS>':2})

            #UNK token for words not seen in training
            if(only_train_words):
               table.update({'<UNK>':3})

            #Loop over the words in current translation
            for word in translation:
                #Check if word doesnt already exists in dict
                if((word in table.keys()) == False):
                    #Save word w/ index (current length of table)
                    #Note: start from index 4, first indexes if for (pad, sos, eos, unk)
                    table.update({word:len(table)})
                    counter.update({word:1})
                else:
                    counter[word] += 1

    #Blank token for CTC layer is N_classes - 1 (tensorflow assumption)
    if(gloss):
        table_gloss.update({'<BLANK>':len(table_gloss)})

    if(remove_singletons):
        freq = [k for k,v in counter.items() if v > 1]
        #Create new vocab
        table = {}

        #Set pad as first index 0
        table.update({'<PAD>':0})

        #Start-of-sentence, end-of-sentence, unknown tokens
        #table.update({'<SOS>':1})
        #table.update({'<EOS>':2})
        table.update({'<UNK>':3})

        #print(len(freq))
        for elem in freq:
            table.update({elem:len(table)})

    if(save):
        with open(os.path.join('data', name+'.txt'), 'wb') as file:
            file.write(pickle.dumps(table))

        if(gloss):
            with open(os.path.join('data', name+'_gloss.txt'), 'wb') as file:
                file.write(pickle.dumps(table_gloss))

    return table, table_gloss

#######################
###Create lookup table

##SLT##
#Vocab -> 2887 + 4 (pad, unk, sos, eos)
#Freq words -> 1810
#Gloss -> 1084 + 2 (unk, blank/pad)

##SLR##
#vocab -> 1231 + 2 (unk, blank/pad)

path = 'data/PHOENIX-2014-T/annotations/manual/'
#path = 'data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/annotations/manual/'

train_file='train.corpus.csv'
train_file = 'PHOENIX-2014-T.train.corpus.csv'
#table, table_gloss = lookup(path, train_file=train_file)

#with open('data/slt_lookup_gloss.txt', 'rb') as pickle_file:
with open('data/slr_lookup.txt', 'rb') as pickle_file:
   content = pickle.load(pickle_file)

print(content)
