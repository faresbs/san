#Methods for visualization and evaluation
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

#Save learning curve for each experiment
def learning_curve(path_1, path_2, path_3):

    curve_1 = os.path.join(path_1, "learning_curves.npy")
    curve_1 = np.load(curve_1, allow_pickle=True)[()]

    curve_2 = os.path.join(path_2, "learning_curves.npy")
    curve_2 = np.load(curve_2, allow_pickle=True)[()]

    curve_3 = os.path.join(path_3, "learning_curves.npy")
    curve_3 = np.load(curve_3, allow_pickle=True)[()]

    
    word_err_1 = curve_1['wer']
    word_err_2 = curve_2['wer']
    word_err_3 = curve_3['wer']

    epochs = range(len(word_err_1))

    plt.figure()

    plt.plot(epochs, word_err_1, 'g', label='STN')
    plt.plot(epochs, word_err_2, 'r', label='+ Flux de main')
    plt.plot(epochs, word_err_3, 'b', label='+ Masquage Locale et Relatif')


    #plt.title('Word Error Rate %')

    plt.xlabel('itération')
    plt.ylabel('Taux d’erreur de mots ')
    plt.legend()

    plt.savefig("plot_wer.png")

#Save learning curve for each experiment
def learning_curve_slt(path_1, path_2):

    curve_1 = os.path.join(path_1, "learning_curves.npy")
    curve_1 = np.load(curve_1, allow_pickle=True)[()]

    curve_2 = os.path.join(path_2, "learning_curves.npy")
    curve_2 = np.load(curve_2, allow_pickle=True)[()]

    bleu_1 = curve_1['bleu_4']
    bleu_2 = curve_2['bleu_4']

    epochs = range(len(bleu_1))

    plt.figure()

    plt.plot(epochs, bleu_1, 'g', label='STN (codeur-décodeur)')
    plt.plot(epochs, bleu_2, 'r', label='STN Hybride ')

    plt.xlabel('itération')
    plt.ylabel('Bleu-4')
    plt.legend()

    plt.savefig("plot_bleu.png")


path_1 = "EXPERIMENTATIONS/SLR/normal/"
path_2 = "EXPERIMENTATIONS/SLR/hand/"
path_3 = "EXPERIMENTATIONS/SLR/rel-hand/"

learning_curve(path_1, path_2, path_3)


path_1 = "EXPERIMENTATIONS/SLT/normal-ls/"
path_2 = "EXPERIMENTATIONS/SLT/hybrid-ls/"

learning_curve_slt(path_1, path_2)