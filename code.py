from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


import pandas as pd
data = pd.read_csv('train.csv')
data.describe()
data.count()
data.info()
data_array = data.values

from numpy import genfromtxt
my_data = genfromtxt('my_file.csv', delimiter=',')

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]) 


clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0) 
clf = clf.fit(iris.data, iris.target)
clf.predict([[ 5.9,  3,  5.0,  1.8]])

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X)

clf = svm.SVC(kernel='linear')   # Build SVM classifier with linear kernel
clf.fit(X, y)         # Fit the training data X and corresponding labels y
clf.predict([[2, 2]])
# get support vectors
clf.support_vectors_
# get indices of support vectors
clf.support_ 
# The indices are 0 and 1 respectively

# get number of support vectors for each class
clf.n_support_ 
# Each class get 1 support vector

np.linspace(2.0, 3.0, num=5)
# Generate 5 numbers from 2 to 3 (inclusive)

np.linspace(2.0, 3.0, num=5, endpoint=False)
# We want 5 numbers, but we do not want to include endpoint 3

#What if I want to know the step?
np.linspace(2.0, 3.0, num=5, retstep=True)
#retstep : bool, optional
# If True, return both (`samples`, `step`), where `step` is the spacing between samples.2.5*np.random.randn(2, 4)+3

# Run this multiple times
np.random.randn()

# For random samples from :math:N(\mu, \sigma^2), 
# 2-by-4 array of samples from N(3, 6.25), i.e. wih mu=3, sigma=2.5 (6.25=2.5^2)
# sigma * np.random.randn (...)   + mu
2.5*np.random.randn(2, 4) + 3

#Draw samples from the distribution:
# -1:  Lower boundary; 0:  upper boundary , 100 numbers
s = np.random.uniform(-1, 0, 100)

#All values are within the given interval:
np.all(s >= -1)



clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
RFscores = cross_val_score(clf, X, y)
print (RFscores.mean())


clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
scores = cross_val_score(clf, X, y)
print (scores.mean()) 

clf = AdaBoostClassifier(n_estimators=100)  # Build AdaBoost classier with 100 base classifiers
scores = cross_val_score(clf, iris.data, iris.target)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)
print (clf.score(X_test, y_test))

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0).fit(X, y)
print ("The feature importance scores for 10 features")
print (clf.feature_importances_)


# Build 3 individual classifiers 
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

# Voting- ensemble classifier
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#, classification

accuracy_score(iris.target, y_pred)

confusion_matrix(iris.target, y_pred)
# Rows are actual labels, while columns are predicted labels

print ('Confusion_matrix\n', confusion_matrix(iris.target, y_pred))

from sklearn.metrics import mean_squared_error
print (mean_squared_error(y_test, est.predict(X_test))) 



from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)

from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_true, y_pred)  


def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'


def title_map(title):
    if title in ['Mr']:
        return 1
    elif title in ['Master']:
        return 3
    elif title in ['Ms','Mlle','Miss']:
        return 4
    elif title in ['Mme','Mrs']:
        return 5
    else:
        return 2

titanic_df['title'] = titanic_df["Name"].apply(get_title).apply(title_map)

X_train, X_test, y_train, y_test = train_test_split(
	iris_X, iris_y, test_size = 0.3)

# row of array
a = [[1,2,3,4,5]]
# column of array
b = [1,2,3,4,5]
