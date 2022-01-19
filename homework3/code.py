from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold, cross_val_predict
import matplotlib.pyplot as plt
from matplotlib import pylab
from scipy.io import arff
import numpy as np

Kfol = KFold(n_splits=5, random_state=0, shuffle=True)

#*2
file = open("breast.w.arff", "r")
data, meta = arff.loadarff(file)

input = data[meta.names()[:-1]].tolist()
output = data["Class"].tolist()

for earlyStop in [False, True]:
    classifier = MLPClassifier(hidden_layer_sizes=(3,2), activation='relu', alpha=0.2, max_iter=2000, early_stopping=earlyStop)
    prevision = cross_val_predict(classifier, input, output, cv=Kfol)
    conf_mat = confusion_matrix(output, prevision)
    disp  = ConfusionMatrixDisplay(conf_mat, display_labels=['Benign','Malignant'])
    disp.plot()
    if earlyStop:
        plt.title("Early Stopping")
    else:
        plt.title("No Early Stopping")

plt.show()

#*3
file.close()
file = open("kin8nm.arff", "r")
data, meta = arff.loadarff(file)

input = data[meta.names()[:-1]].tolist()
output = data["y"].tolist()

for alpha, graph in zip([0, 0.2], [1,2]):
    classifier = MLPRegressor(hidden_layer_sizes=(3,2), activation='relu', alpha=alpha, max_iter=2000)
    classifier.fit(input, output)
    prevision = cross_val_predict(classifier, input, output, cv=Kfol)
    residues = np.subtract(output, prevision)
    plt.subplot(1, 2, graph)
    plt.boxplot(residues)
    if graph == 1:
        plt.title("No Regularization")
    else:
        plt.title("With Regularization")

plt.show()