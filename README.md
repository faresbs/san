# Sign Language Recognition and Translation 

This repository provides a pytorch-based implementation of Context Matters: Self-Attention for Sign Language Recognition.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Updates

* Paper submitted into ICPR 2020.
  

### Prerequisites

Of course, you need to have python, here we are using python 3.6. So you need to install python3.

```
sudo apt-get update
sudo apt-get install python3.6
```

Install Pytorch a very cool machine learning library and the CUDA environment. 

```
https://pytorch.org/
```

Install opencv2.
```
sudo apt-get install python-opencv
```

Other dependencies (numpy, etc..).
```
pip install numpy
```


## Architecture

<img src="https://github.com/faresbs/slrt/blob/master/images/arch.png" width="800" height="600" />

## Evaluation 
To train the Transformer model for SLR (Sign Language Recognition)
```
python evalaute_slr.py
```
To train the Transformer model for SLT (Sign Language Translation)
```
python evalaute_slr.py
```

### Text simplification
After generating the prediction/translation output texts for the whole text, you can use the above script to remove the unwanted tokens like stop words (This will improve recognition performance).
```
./simplify.sh <path of the generated texts>
```

## Training
To train the Transformer model for SLR (Sign Language Recognition)
```
python train_slr.py
```
To train the Transformer model for SLT (Sign Language Translation)
```
python train_slt.py
```

## Built With

* [Pytorch](https://pytorch.org/) - ML library
* [Opencv](https://opencv.org/) - Open Source Computer Vision Library

## TO DO:

## Results

### Quantitative Analysis

<img src="https://github.com/faresbs/slrt/blob/master/images/table.png" width="300" height="400" />

### Qualitative Analysis

<img src="https://github.com/faresbs/slrt/blob/master/images/heatmap.png" width="800" height="600" />

## Datasets

### RWTH-PHOENIX-Weather 2014: Continuous Sign Language Recognition Dataset
https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

### RWTH-PHOENIX-Weather 2014 T: Parallel Corpus of Sign Language Video, Gloss and Translation
https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

## Contributing

You are free to use this project or contribute that would be cool. Please contact me if you face any problems running the code or if you require any clarification.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

* **Fares Ben Slimane** - *Grad Student (UQAM)* - [check my personal webpage](http://faresbs.github.io)
* **Mohamed Bouguessa** - *Professor (UQAM)*

## Acknowledgments
* Please check the Github repo (https://github.com/neccam/SubUNets) for the implementation of "SubUNets: End-to-end Hand Shape and Continuous Sign Language Recognition" (ICCV'17).
* 



