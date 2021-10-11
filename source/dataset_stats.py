#OPTIONAL

#Get std and mean of the dataset

"""
We can use ImageNet statistics (std and mean) to normalize the dataset images.
However since we are dealing with a different domain image, then it is better to calculate the statistics of our target dataset.
You can use these stats to normalize data or use the one provided by Camgoz et al. 2017
"""

import torch
import torchvision
import progressbar

def get_statistics(data_path):

    dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=torchvision.transforms.ToTensor()
        )

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            shuffle=False
        )

    print("Dataset size: "+str(len(dataset)))

    #For progress bar
    bar = progressbar.ProgressBar(maxval=5000, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    i = 0

    for batch_idx, (data, target) in enumerate(loader):

        if(batch_idx == 5000):
            break

        #Update progress bar with every iter
        i += 1
        bar.update(i)

        #Loop over the images
        mean = 0.
        std = 0.
        nb_samples = 0.

        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean, std


#Measure dataset stats (mean/std) for each dataset

#SLT
#data_path = 'data/PHOENIX-2014-T/features/fullFrame-210x260px/train/'

#SLR
data_path = 'data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/'
#data_path = 'data/phoenix-2014.v3/phoenix2014-release/phoenix-2014-multisigner/features/trackedRightHand-92x132px/train'

#Loop over the dataset to measure mean and std
#NOTE: this will probably take a while (run just one time!)
#mean, std = get_statistics(data_path)

##SLT
#mean = torch.tensor([0.5066, 0.3351, 0.1024])
#std = torch.tensor([0.1395, 0.1087, 0.0785])

#Image mean: tensor([0.4981, 0.4894, 0.4675])
#Image std: tensor([0.2928, 0.2938, 0.3329])

##SLR
#FULL IMAGE
#Image mean: tensor([0.5876, 0.5686, 0.5651])
#Image std: tensor([0.3348, 0.3421, 0.3627])

#HAND IMAGE
#Image mean: tensor([0.3843, 0.3587, 0.3459])
#Image std: tensor([0.3285, 0.3154, 0.3291])

mean= torch.tensor([0.5876, 0.5686, 0.5651])
std = torch.tensor([0.3348, 0.3421, 0.3627])

print("Image mean: "+str(mean))
print("Image std: "+str(std))

stat = {'mean':mean, 'std':std}

#Save stats
torch.save(stat, 'data_stats.pt')
#torch.save(stat, 'hand_stats.pt')
