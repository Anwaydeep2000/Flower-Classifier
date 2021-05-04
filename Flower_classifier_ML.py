#Loading required modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

#loading Data set
iris= datasets.load_iris()

#Printing the descriction and feaures
#print(iris.DESCR)
features = iris.data
labels = iris.target


#Training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)

preds = clf.predict([[31,1,1,1]])
print((preds))