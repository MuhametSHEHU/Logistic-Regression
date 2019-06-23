#Load Dataset
from sklearn.datasets import load_iris
iris=load_iris()
print(iris)
#Features
print(iris.feature_names)
#Targets
print(iris.target_names)
# Type of IrisDataset
print(type(iris.data))
# Type of target
print(type(iris.target))
X = iris.data
y = iris.target
from sklearn.linear_model import LogisticRegression
logregression = LogisticRegression()
logregression.fit(X, y)
prediction_lr = logregression.predict([[2,4,3,1], [4,6,5,3]])
print(prediction_lr)
"""
[2 2]
"""
print(iris.target_names)
"""
['setosa' 'versicolor' 'virginica']
"""
prediction1 = logregression.predict([[5.9, 3. , 5.1, 1.8], [4.7, 3.2, 1.3, 0.2], [2, 4, 3, 1]])
print(prediction1)
"""
[2 0 2]
"""