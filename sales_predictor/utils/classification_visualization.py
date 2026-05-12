import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm):

    plt.figure(figsize=(6,6))

    plt.imshow(cm)

    plt.colorbar()

    plt.xticks([0,1], ['Pred 0', 'Pred 1'])
    plt.yticks([0,1], ['True 0', 'True 1'])

    for i in range(2):
        for j in range(2):
            plt.text(j, i,
                     cm[i,j],
                     ha='center',
                     va='center')

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("/outputs/confusion_matrix.png")

    plt.show()

def probability_distribution(probabilities):

    plt.figure(figsize=(10,6))

    plt.hist(probabilities, bins=20)

    plt.xlabel("Predicted Probability")

    plt.ylabel("Frequency")

    plt.title("Probability Distribution")

    plt.grid()

    plt.tight_layout()

    plt.savefig("/outputs/probability_distribution.png")

    plt.show()

