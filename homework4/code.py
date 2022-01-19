from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.io import arff

file = open("breast.w.arff", "r")
data, meta = arff.loadarff(file)

input = data[meta.names()[:-1]].tolist()
output = data["Class"].tolist()

#*4)
kMeans2 = KMeans(n_clusters=2, random_state=0).fit(input)
kMeans3 = KMeans(n_clusters=3, random_state=0).fit(input)
labels2 = kMeans2.labels_
labels3 = kMeans3.labels_

def ECR(labels, target, nLabels):    
    clusters = []
    for i in range(nLabels):
        clusters += [[0,0]]

    for i in range(len(labels)):
        if target[i]==b'benign':
            clusters[labels[i]][0] += 1
        elif target[i]==b'malignant':
            clusters[labels[i]][1] += 1

    sum = 0
    for i in clusters:
        sum += (i[0] + i[1]) - max(i[0],i[1])

    return sum/nLabels

print("ECR k=2:", ECR(labels2, output, 2))
print("ECR k=3:", ECR(labels3, output, 3))

print("Silhouette k=2:", silhouette_score(input, labels2))
print("Silhouette k=3:", silhouette_score(input, labels3))

#*5)
n=3
selector = SelectKBest(mutual_info_classif, k=2)
kBest = selector.fit_transform(input,output)

features = selector.get_support(indices=True)
labels = kMeans3.labels_
centroids = kMeans3.cluster_centers_
print(centroids)

for c in range(n):
    cluster = [kBest[i] for i in range(len(labels)) if labels[i] == c]
    label = "Cluster {}".format(c)
    plt.scatter([x[0] for x in cluster],[y[1] for y in cluster], alpha=0.25, label=label)

plt.scatter([x[features[0]] for x in centroids],[y[features[1]] for y in centroids],marker='X', c='black', label='Centroids')
plt.xlabel(meta.names()[features[0]].replace("_"," "))
plt.ylabel(meta.names()[features[1]].replace("_"," "))
plt.legend()
plt.show()

















