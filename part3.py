#Using RBF Kernel/algorithm
import imp
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris=datasets.load_iris()#loading the iris dataset
#spliting the data into test and training set
xtrain,xtest,ytrain,ytest=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
#using rbf algorithmn kernel type  for classification,with error penalty of 1
my_classification=svm.SVC(kernel='rbf',C=1).fit(xtrain,ytrain)#rbf is the default value
print(my_classification.score(xtest,ytest))#printing accuracy level

#linear Kernel has a better accuracy value given the same test_size:LKernel=96.67% while the RBF has 95%
