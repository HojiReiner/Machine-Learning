from sklearn import tree
from scipy.io import arff
import numpy as num
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_validate, KFold

file = open("breast.w.arff", "r")
data, meta = arff.loadarff(file)

input = data[meta.names()[:-1]].tolist()
output = data["Class"].tolist()

print("5)")
print("i.")
slFeatures = [[],[]]
kFol = KFold(n_splits=10, shuffle=True, random_state=47)
for i in [1,3,5,9]:
    input_new = SelectKBest(mutual_info_classif, k=i).fit_transform(input,output)
    classifier = tree.DecisionTreeClassifier(criterion='entropy')
    crossRes = cross_validate(classifier, input_new, output, scoring = 'accuracy', cv = kFol, return_train_score=True)
    slFeatures[0].append(num.average(crossRes['test_score']))
    slFeatures[1].append(num.average(crossRes['train_score']))

print("Testing Accuracy:", slFeatures[0])
print("Training Accuracy:", slFeatures[1])
print("ii.")
slDepth=[[],[]]
for i in [1,3,5,9]:
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    crossRes = cross_validate(classifier, input, output, scoring = 'accuracy', cv = kFol, return_train_score=True)
    slDepth[0].append(num.average(crossRes['test_score']))
    slDepth[1].append(num.average(crossRes['train_score']))

print("Testing Accuracy:", slDepth[0])
print("Training Accuracy:", slDepth[1])

plt.subplot(1,2,1)
plt.plot([1,3,5,9], slFeatures[0], 'b-', label="Test Score")
plt.plot([1,3,5,9], slFeatures[1], 'y-', label="Train Score")
plt.title("Selected Features")
plt.legend()
plt.subplot(1,2,2)
plt.plot([1,3,5,9], slDepth[0], 'b-')
plt.plot([1,3,5,9], slDepth[1], 'y-')
plt.title("Max Depth")
plt.show()

""" 
#deepth = []
for i in [1,3,5,9]:
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    cross = cross_val_score(classifier, input, output, scoring = 'accuracy', cv = kFol)
    print(i, "deepth:", num.average(cross))
    #deepth += [num.average(cross)] """

""" plt.plot(x,selected, color='r')
plt.plot(x,deepth, color='g')
plt.show() """
""" 
print("training:")
x = [1,3,5,9]
selected = []



print("ii.")
deepth = []
for i in x:
    tree1 = tree.DecisionTreeClassifier(max_depth=i)
    tree1.fit(obs,target)
    cross = tree1.predict(obs)
    correct = 0
    for j in range(len(target)):
        if cross[j] == target[j]:
            correct+=1
    print(i, "deepth:", correct/len(target))
    deepth += [correct/len(target)]

plt.plot(x,selected, color='r')
plt.plot(x,deepth, color='g')
plt.show() """








