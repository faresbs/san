#!/usr/bin/python
from PIL import Image
import os, sys
import argparse
import progressbar
import shutil

def resize(path, resize=227):

    dirs = os.listdir(path)

    #For progress bar
    bar = progressbar.ProgressBar(maxval=len(dirs), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    cnt_bar = 0

    for folder in dirs:

        cnt_bar += 1
        bar.update(cnt_bar)

        #Loop over images in subfolder
        images = os.listdir(os.path.join(path, folder))
        for img in images:
            img = os.path.join(path, folder, img)

            if os.path.isfile(img):
                try:
                    im = Image.open(img)
                    #f, e = os.path.splitext(path+item)
                    #im.verify()
                    im = im.resize((resize, resize), Image.ANTIALIAS)
                    im.save(img)

                except (IOError, SyntaxError) as e:
                    print('error reading image '+img)
                    quit(0)


##############
###Arg parsing

parser = argparse.ArgumentParser(description='Resize images from dataset')

#parser.add_argument('--path', type='str', default=('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/keyfeatures/fullFrame-210x260px',
#                    help='Dataset path to resize')

parser.add_argument('--size', type=int, default=227,
                    help='size to resize the images')

parser.add_argument('--copy_path', type=bool, default=None,
                    help='copy the dataset to copy_path and resize or simply resize the same dataset')


args = parser.parse_args()

#Loop over all 3 datasets
datasets = ['train']

#if(args.copy_path):
#    print('Copying dataset..')
#    shutil.copytree(args.copy_path, args.copy_path)

for dataset in datasets:
    print("Resizing "+dataset)

    #path = os.path.join(args.path, dataset)

    #path = os.path.join("data/PHOENIX-2014-T/keyfeatures/fullFrame-210x260px", dataset)

    path = os.path.join('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/keyfeatures/fullFrame-210x260px', dataset)
    #path_hand = os.path.join('data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/keyfeatures/trackedRightHand-92x132px', dataset)

    resize(path, resize=args.size)
    print("DONE!")
    #resize(path_hand, resize=args.size)
    #print("DONE!")
