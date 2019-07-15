import pandas as pd

kesuburan = pd.read_csv('fertility/fertility.csv')


# age, accident/trauma, sitting hour, freq alcohol, smoking habit
categorical = ['Childish diseases', 'Accident or serious trauma', 'Surgical intervention', 'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking habit']
x = kesuburan.drop(['Diagnosis', 'Season'], axis=1)
y = kesuburan['Diagnosis']

categorical = x.drop(['Age', 'Number of hours spent sitting per day'], axis=1).columns

for feature in categorical:
    x[feature].fillna('missing', inplace=True)
    dummies = pd.get_dummies(x[feature], prefix=feature)
    x = pd.concat([x, dummies], axis=1)
    x.drop([feature], axis=1, inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
kesuburan['Diagnosis'] = le.fit_transform(kesuburan['Diagnosis'])
y = kesuburan['Diagnosis']
diagnosis = le.classes_

# TST
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

# modelling
# logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
logreg = LogisticRegression(multi_class='auto', solver='lbfgs')
logreg.fit(xtrain, ytrain)
yLogreg = logreg.predict(xtest)
logreg_score = accuracy_score(ytest, yLogreg)

# support vector classifier
from sklearn.svm import SVC
svc = SVC(gamma='auto')
svc.fit(xtrain, ytrain)
ySvc = svc.predict(xtest)
svc_score = accuracy_score(ytest, ySvc)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(xtrain, ytrain)
yknn = knn.predict(xtest)
knn_score = accuracy_score(ytest, yknn)

childDis_n = [1, 0]
childDis_y = [0, 1]
accident_n = [1, 0]
accident_y = [0, 1]
surgical_n = [1, 0]
surgical_y = [0, 1]
fever_less = [1, 0, 0]
fever_more = [0, 1, 0]
fever_no = [0, 0, 1]
alcohol_everyday = [1, 0, 0, 0, 0]
alcohol_hardly = [0, 1, 0, 0, 0]
alcohol_onceWeek = [0, 0, 1, 0, 0]
alcohol_severalDay = [0, 0, 0, 1, 0]
alcohol_severalWeek = [0, 0, 0, 0, 1]
smoking_daily = [1, 0, 0]
smoking_never = [0, 1, 0]
smoking_occasional = [0, 0, 1]

# out of sample data
# age, hour sit, child dis, accident, surgical, fever, alcohol, smoking
sample = [
    ('arin' , [29, 5] + childDis_n + accident_n + surgical_n + fever_no + alcohol_everyday + smoking_daily),
    ('bebi' , [31, 24] + childDis_n + accident_y + surgical_y + fever_no + alcohol_severalWeek + smoking_never),
    ('caca' , [25, 7] + childDis_y + accident_n + surgical_n + fever_more + alcohol_hardly + smoking_never),
    ('dini' , [28, 24] + childDis_n + accident_y + surgical_y + fever_no + alcohol_hardly + smoking_daily),
    ('enno' , [42, 8] + childDis_y + accident_n + surgical_n +fever_no + alcohol_hardly + smoking_never)
]
for key, value in sample:
    ylogreg = logreg.predict([value])
    ysvc = svc.predict([value])
    yknn = knn.predict([value])
    print(f'{key}, kesuburan: {diagnosis[ylogreg]} (Logistic Regression)')
    print(f'{key}, kesuburan: {diagnosis[ysvc]} (Support Vector Classifier)')
    print(f'{key}, kesuburan: {diagnosis[yknn]} (K-Nearest Neigbors)')
