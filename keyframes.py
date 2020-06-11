"""
script to extract keyframes from sequence of images using ffmpeg library
-> reduce sequence length for faster training
-> discard irrelevant/repetitive frames
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import argparse
import subprocess
import progressbar

import parser

#video editing library
import ffmpeg


#Can work for a single sequence of frames or multiple sequence frames
def data_extraction(path, save_path, seq_length):


    #Check if path folder contains subfolders
    list_dir = os.listdir(path)
    for f in list_dir:
        if not os.path.isfile(os.path.join(path, f)):
            contains_subfolders = True
            #No need to loop over the rest of subfolders
            break

        print("No subfolders found!")
        quit(0)

    #Loop over the sub folders
    if(contains_subfolders):

        #Progress Bar
        bar = progressbar.ProgressBar(maxval=len(list_dirs), widget=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        cnt_bar = 0

        #Note: each subfolder contains a sequence of frames
        for subfolder in list_dir:

            cnt_bar += 1
            bar.update(cnt_bar)

            subfolder_path = os.path.join(path, subfolder)

            list_frames = os.listdir(subfolder_path)

            #Create new folder for current subfolder w/ same name
            new_path = os.path.join(save_path, subfolder)

            if(os.path.exists(new_path)==False):
                os.mkdir(new_path)

            #Number of frames in sequence
            len_frames = len(list_frames)

            #Pick thumbnails if number of frames is bigger than seq_len
            #Note: we may get number of keyframes less than seq_len
            if(len_frames > seq_length):

                #Calculate number of thumbnails to be used
                #selects one representative frame from each set of n frames
                #Reduce to number of seq_length or less depending on the clip

                t = math.ceil(len_frames / seq_length)

                inFile = os.path.join(subfolder_path, 'images%04d.png')
                outFile = os.path.join(new_path, 'images%04d.png')

                #Extract keyframes from video
                cmd = 'ffmpeg -i '+inFile+' -vf thumbnail='+str(t)+',setpts=N/TB -r 1 -vframes '+str(seq_length)+' '+outFile
                #Run Command
                subprocess.call(cmd, shell=True)

            else:
                #Save keyframes in new path
                for k in range(1, len_frames):

                    frame = os.path.join(subfolder_path, 'images{:04d}'.format(k)+'.png')
                    frame = plt.imread(frame)
                    plt.imsave(os.path.join(new_path, 'images{:04d}'.format(k)+'.png'), frame)


#Use this to replace weird existing naming format with a better one
def replace(path, path_hand):

    #Check if path folder contains subfolders 
    list_dir = os.listdir(path)

    for f in list_dir:
        if not os.path.isfile(os.path.join(path, f)):
            contains_subfolders = True
            #No need to loop over the rest of subfolders
            break

        print("No subfolders found!")
        quit(0)

    #Loop over the sub folders
    if(contains_subfolders):

        for subfolder in list_dir:

            subfolder_path = os.path.join(path, subfolder, '1')
            subfolder_path_hand = os.path.join(path_hand, subfolder, '1')

            list_frames = os.listdir(subfolder_path)
            list_frames.sort()

            #Copy images to destination with new name
            [os.rename(os.path.join(subfolder_path, frame), os.path.join(subfolder_path, "images{:04d}.png".format(i))) for i, frame in enumerate(list_frames)]

            list_frames_hand = os.listdir(subfolder_path_hand)
            list_frames_hand.sort()

            #Copy images to destination with new name
            [os.rename(os.path.join(subfolder_path_hand, frame), os.path.join(subfolder_path_hand, "images{:04d}.png".format(i))) for i, frame in enumerate(list_frames_hand)]


#Extract keyframes from train, val, test datasets and save
if __name__ == "__main__":

    parser.add_argument('--path', type='str', default='data/PHOENIX-2014-T/features/fullFrame-210x260px',
                        help='Source dataset path')

    parser.add_argument('--save_path', type='str', default='data/PHOENIX-2014-T/keyfeatures/fullFrame-210x260px',
                        help='Destination dataset path')

    parser.add_argument('--keyframes', type=int, default=64,
                        help='Extract N keyframes from source sequence of frames')


    #Reduce clip length to 64 or less
    seq_length = args.keyframes

    datasets = ['train']

    for dataset in datasets:

        save_path = os.path.join(args.save_path, dataset)

        if(os.path.exists(save_path)==False):
            os.makedirs(save_path)

        path = os.path.join(args.path, dataset)

        data_extraction(path, save_path, seq_length)
































