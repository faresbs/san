#Methods for visualization and evaluation
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

#Save learning curve for each experiment
def learning_curve_slr(path):

    curve_path = os.path.join(path, "learning_curves.npy")
    curve = np.load(curve_path, allow_pickle=True)[()]

    train_ppls = curve['train_ppls']
    val_ppls = curve['val_ppls']
    word_err = curve['wer']
    epochs = range(len(train_ppls))

    plt.figure()

    plt.plot(epochs, train_ppls, 'b', label='train perplexity')
    plt.plot(epochs, val_ppls, 'g', label='val perplexity')
    plt.title('Train and val Perplexity')
    plt.xlabel('epoch')
    plt.legend()

    #plt.show()
    #Save with best loss + epoch
    plt.savefig(os.path.join(path,"plot_loss.png"))

    plt.figure()

    plt.plot(epochs, word_err, 'g', label='WER')
    plt.title('Word Error Rate %')

    plt.xlabel('epoch')
    plt.legend()

    plt.savefig(os.path.join(path,"plot_wer.png"))


def learning_curve_slt(path):

    curve_path = os.path.join(path, "learning_curves.npy")
    curve = np.load(curve_path, allow_pickle=True)[()]

    train_ppls = curve['train_ppls']
    val_ppls = curve['val_ppls']
    bleu_4 = curve['bleu_4']
    epochs = range(len(train_ppls))

    plt.figure()

    plt.plot(epochs, train_ppls, 'b', label='train perplexity')
    plt.plot(epochs, val_ppls, 'g', label='val perplexity')
    plt.title('Train and val Perplexity')
    plt.xlabel('epoch')
    plt.legend()

    #plt.show()
    #Save with best loss + epoch
    plt.savefig(os.path.join(path,"plot_loss.png"))

    plt.figure()

    plt.plot(epochs, val_ppls, 'b', label='bleu-4')
    plt.title('Bleu score')

    plt.xlabel('epoch')
    plt.legend()

    plt.savefig(os.path.join(path,"plot_bleu.png"))



#if __name__ == "__main__":
#    if len(sys.argv) < 1:
#        print('Usage: utils.py <learning curve path>')
#        sys.exit(0)

#    curve_path = sys.argv[1]

    #curve = np.load(curve_path, allow_pickle=True)[()]
#    learning_curve(curve_path)
