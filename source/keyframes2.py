"""
script to extract keyframes from sequence of images using ffmpeg library
-> reduce sequence length for faster training
-> discard irrelevant/repetitive frames
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import math
import shutil
import glob
import subprocess
import progressbar
import _pickle as pickle
#video editing library
import ffmpeg



def keyframes_wassim(path, save_path, seq_length):
    skipped=0
    with open('wordFrameLabelPerVideo.pickle', 'rb') as pickle_file:
        frame_label = pickle.load(pickle_file)
        #keyframe_label = {}
        list_dir = os.listdir(path)
        for f in list_dir:
            if not os.path.isfile(os.path.join(path,f)):
                contains_subfolders = True
                break
            print("No subfolders found!")
            quit(0)
        if(contains_subfolders):
            for subfolder in list_dir:
                #if (subfolder not in frame_label):
                #    skipped +=1
                #    continue
                #keyframe_label[subfolder]=[]
                subfolder_path = os.path.join(path, subfolder)
                print(subfolder_path)
                list_frames = os.listdir(subfolder_path)
                new_path = os.path.join(save_path, subfolder)
                if(os.path.exists(new_path)==False):
                    os.makedirs(new_path)
                len_frames = len(list_frames)
                if(len_frames > seq_length):
                    step= len_frames/seq_length
                    next_frame=0
                    number=0
                    frame = os.path.join(subfolder_path,'images{:04d}'.format(next_frame)+'.png')
                    frame = plt.imread(frame)
                    plt.imsave(os.path.join(new_path, 'images{:04d}'.format(next_frame)+'.png'), frame)
                    #keyframe_label[subfolder].append(frame_label[subfolder][next_frame])
                    for i in range(1,seq_length):
                        next_frame += step
                        number +=1
                        img_num=math.floor(next_frame)
                        frame = os.path.join(subfolder_path,'images{:04d}'.format(img_num)+'.png')
                        frame = plt.imread(frame)
                        plt.imsave(os.path.join(new_path, 'images{:04d}'.format(number)+'.png'), frame)
                        #keyframe_label[subfolder].append(frame_label[subfolder][img_num])
                else:
                    for k in range(1, len_frames):
                        frame = os.path.join(subfolder_path, 'images{:04d}'.format(k)+'.png')
                        frame = plt.imread(frame)
                        plt.imsave(os.path.join(new_path, 'images{:04d}'.format(k)+'.png'), frame)
                        #keyframe_label[subfolder].append(frame_label[subfolder][k])
            #file_to_write = open("KeyframeLabelPerVideo_wassim.pickle", "wb")
            #pickle.dump(keyframe_label, file_to_write)
            #file_to_write.close()
            print(skipped)
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

        #Progress bar
        #bar = progressbar.Progress

        #Note: each subfolder contains a sequence of frames
        for subfolder in list_dir:

            subfolder_path = os.path.join(path, subfolder)
            #subfolder_path_hand = os.path.join(path_hand, subfolder)
            print(subfolder_path)
            list_frames = os.listdir(subfolder_path)

            #Create new folder for current subfolder w/ same name
            new_path = os.path.join(save_path, subfolder)

            if(os.path.exists(new_path)==False):
                os.makedirs(new_path)

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
                #NOTE: encoding on average 1-3 images
                cmd = 'ffmpeg -i '+inFile+' -vf thumbnail='+str(t)+',setpts=N/TB -r 1 -vframes '+str(seq_length)+' '+outFile
                #Run Command without bash output
                subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            else:

                #Save keyframes in new path
                for k in range(1, len_frames):

                    frame = os.path.join(subfolder_path, 'images{:04d}'.format(k)+'.png')
                    frame = plt.imread(frame)
                    plt.imsave(os.path.join(new_path, 'images{:04d}'.format(k)+'.png'), frame)

#Use this to replace crapy existing naming format with a better one
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

            if(os.path.exists(os.path.join(path, subfolder, '1'))):
                src_path = os.path.join(path, subfolder, '1')

                dest_path = os.path.join(path, subfolder)

                list_frames = os.listdir(src_path)
                list_frames.sort()

                #Copy images to destination with new name
                [shutil.move(os.path.join(src_path, frame), os.path.join(dest_path, "images{:04d}.png".format(i))) for i, frame in enumerate(list_frames)]

                if len(os.listdir(src_path)) == 0: # Check if old folder is empty
                    shutil.rmtree(src_path) # Delete old folder

            if(os.path.exists(os.path.join(path_hand, subfolder, '1'))):
                src_path_hand = os.path.join(path_hand, subfolder, '1')

                dest_path_hand = os.path.join(path_hand, subfolder)

                list_frames_hand = os.listdir(src_path_hand)
                list_frames_hand.sort()

                #Copy hand images to destination with new name
                [os.rename(os.path.join(src_path_hand, frame), os.path.join(dest_path_hand, "images{:04d}.png".format(i))) for i, frame in enumerate(list_frames_hand)]

                if len(os.listdir(src_path_hand)) == 0: # Check if old folder is empty
                    shutil.rmtree(src_path_hand) # Delete old folder



#Extract keyframes from train, val, test datasets and save 
if __name__ == "__main__":

    #Reduce clip length to 64 or less
    seq_length = 64

    datasets = ['dev']

    print('This will take a while..')

    for dataset in datasets:

        save_path = os.path.join('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/keyfeatures_wassim/fullFrame-210x260px', dataset)
        #save_path_hand = os.path.join('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/keyfeatures_wassim/trackedRightHand-92x132px', dataset)

        path = os.path.join('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px', dataset)
        path_hand = os.path.join('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/trackedRightHand-92x132px', dataset)

        #path = os.path.join('data/PHOENIX-2014-T/features/fullFrame-210x260px', datasets)
        #save_path = os.path.join('data/PHOENIX-2014-T/keyfeatures/fullFrame-210x260px', datasets)

        #Replace naming format
        #NOTE: Do this once
        #print("Replacing format "+dataset+'..')
        #replace(path, path_hand)

        #print('Extracting '+dataset+'..')
        #Extract Key frames
        #data_extraction(path, save_path, seq_length)
        #data_extraction(path_hand, save_path_hand, seq_length)

        keyframes_wassim(path, save_path, seq_length)
































