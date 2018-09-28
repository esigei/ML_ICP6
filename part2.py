#Implement Linear SVM using scikit-lib
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris=datasets.load_iris()#loading the iris dataset
#spliting the data into test and training set
xtrain,xtest,ytrain,ytest=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
#using linear algorithmn kernel type  for classification,with error penalty of 1
my_classification=svm.SVC(kernel='linear',C=1).fit(xtrain,ytrain)
print("Linear Kernel: ",my_classification.score(xtest,ytest))#printing accuracy level

# the linear svm has a better accuracy than naive bayes; naiveB=95% while the LinearSVM=96.67%