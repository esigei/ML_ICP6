from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
#loading the iris data set
iris=datasets.load_iris()
#spliting the dataset into training and test set using 40% portion of the data as test and 60% training
xtrain,xtest,ytrain,ytest=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
#creating an instance of Gausian Naive bayes
gnb=GaussianNB()
#using the test set to fit the gausian model
gnb.fit(xtrain,ytrain)
expected_outcomes=ytest #using the y test set as the expected outcomes
prediction=gnb.predict(xtest) #using the xtest set to make prediction of the outcomes
#print(metrics.classification_report(expected_outcomes,prediction)) #printing the classification results based on Gausian Naive Bayes
#other way to get accuracy
print("Accuracy: ",metrics.accuracy_score(expected_outcomes,prediction))
#print("Accuracy of predic: ",metrics.accuracy_score(prediction,expected_outcomes))
#gives a precision percentage of 93.333%