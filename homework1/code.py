import matplotlib.pyplot as plt
import numpy as num
from scipy.io import arff
from scipy.stats import ttest_rel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

file = open("breast.w.arff", "r")
data, meta = arff.loadarff(file)

#* Class conditional distribution
bins = [1,2,3,4,5,6,7,8,9,10,11]
for features, i in zip(meta.names()[:-1], range(1,10)):
    plt.subplot(3,3,i)
    d = [data[(data["Class"]==b'benign')][features] ,data[(data["Class"]==b'malignant')][features]]
    plt.hist(d,bins=bins,align='left',color=['g','r'], label=['Benign','Malignant'],alpha=0.6,histtype='stepfilled', density=True)
    if i==3: plt.legend()
    plt.title(features.replace("_"," "))

plt.show()

#* 10-fold cross-validation knn accuracy for k = 3,5,7
input = data[meta.names()[:-1]].tolist()
output = data["Class"].tolist()
kFol = KFold(n_splits=10, shuffle=True, random_state=47)
crossKNN=[[],[],[]]

for i,k in zip(range(3,8,2),range(3)):
    classifier = KNeighborsClassifier(n_neighbors=i, weights='uniform', metric='euclidean')
    crossKNN[k] = cross_val_score(classifier, input, output, scoring='accuracy', cv = kFol)
    kErr = num.average(crossKNN[k]) #media dos erros obtidos para k = i
    print("Accuracy K={}: {}".format(i,kErr))

#* Compara Naive Bayes with knn with k = 3
classifier = MultinomialNB()
crossNB = cross_val_score(classifier, input, output, scoring='accuracy', cv = kFol)
pval = ttest_rel(crossKNN[0], crossNB, alternative='greater').pvalue
print("p-value:",pval)