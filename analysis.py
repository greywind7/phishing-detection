import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

phishing = pd.read_csv('dataset/phishing.csv').dropna() * 1
phishing = phishing.astype({'Encoding':'string','HTML content':'int'})
clean = pd.read_csv('dataset/clean.csv').dropna() * 1
clean = clean.astype({'Encoding':'string','HTML content':'int'})
df =  pd.concat([phishing,clean])

phishy_ban = phishing['BanWords'].values
clean_ban = clean['BanWords'].values
plt.hist([phishy_ban,clean_ban])
plt.show()
plt.savefig('figures/Banwords.png')

i = 0
for col in df.columns:
    print(i,col)
    i+=1

print(df.describe())

df = df.drop(df.columns[[0,2,4,6,5,12]], axis = 1)
ph = df.pop('Phishy')
df['Phishy'] = ph
print(df)

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
(x_train,x_test, y_train,y_test) = train_test_split(x,y,test_size = 0.25, random_state = 0)

# Feature scaling is must in geometry based algorithms
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

svm_classifier = LinearSVC(random_state=0)
svm_classifier.fit(x_train,y_train)

ker_classifier = SVC(kernel='rbf', random_state=0)
ker_classifier.fit(x_train,y_train)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(x_train,y_train)

lr_classifier = LogisticRegression()
lr_classifier.fit(x_train,y_train)

nb_classifier = GaussianNB()
nb_classifier.fit(x_train,y_train)

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_classifier.fit(x_train,y_train)

rf_classifier = RandomForestClassifier(random_state=0, criterion='entropy', n_estimators=10)
rf_classifier.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score

print("Logistic Regression")
y_pred = lr_classifier.predict(sc.transform(x_test))
print(confusion_matrix(y_test,y_pred))
sbn.heatmap(confusion_matrix(y_test,y_pred))
plt.show
print(accuracy_score(y_test, y_pred))
print()

y_pred = nb_classifier.predict(sc.transform(x_test))
print("Naive Bayes")
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print()

print("Decision Tree")
y_pred = dt_classifier.predict(sc.transform(x_test))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print()

print("Random Forest")
y_pred = rf_classifier.predict(sc.transform(x_test))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print()

print("Linear SVM")
y_pred = svm_classifier.predict(sc.transform(x_test))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print()

print("Kernel SVM")
y_pred = ker_classifier.predict(sc.transform(x_test))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print()


