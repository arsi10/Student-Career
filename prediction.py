import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Waiting for user response
warnings.filterwarnings('ignore')

print('Press Enter to load dataset')
input()
data = pd.read_excel('Pred/Input/TrainData.xlsx')

print('Press Enter to preprocess dataset')
input()

# Data Preprocessing includes removing irrelevant columns and changing categorical data into numerical
def preprocess(data):
    data = data.drop(columns=['Quantity', 'Price Tier', 'Ticket Type','Attendee #', 'Group', 'Order Type', 'Currency', 'Total Paid','Fees Paid', 'Eventbrite Fees', 'Eventbrite Payment Processing','Attendee Status', 'College Name','How did you come to know about this event?','Specify in "Others" (how did you come to know about this event)','Designation', 'Year of Graduation'])
    return data


data = preprocess(data)
data['Placement Status'] = data['Placement Status'].astype('category')
data['Placement Status'] = data['Placement Status'].map({'Placed' : 1,'Not placed' : 0})
print('After Preprocessing, dataset looks like this: ')
data.head()
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

# Assigning independent columns to X and dependent column to Y
X = data.drop(columns=['First Name','Email ID','Placement Status'],axis=1)
Y = data['Placement Status']

print('Press Enter to start training and testing model')
input()
# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Using Logistic Regression Algorithm
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Using SVM Algorithm
svm = svm.SVC()
svm.fit(X_train, y_train)

# Using K-Neighbors Classifier Algorithm
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Using Decision Tree Classifier Algorithm
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Using Random Forest Classifier Algorithm
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Using Gradient Boosting Classifier Algorithm
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

# Testing the Model
y_pred1 = lr.predict(X_test)
y_pred2 = svm.predict(X_test)
y_pred3 = knn.predict(X_test)
y_pred4 = dt.predict(X_test)
y_pred5 = rf.predict(X_test)
y_pred6 = gb.predict(X_test)

# Calculating accuracy score
lr_score = accuracy_score(y_test,y_pred1)
svm_score = accuracy_score(y_test,y_pred2)
knn_score = accuracy_score(y_test,y_pred3)
dt_score = accuracy_score(y_test,y_pred4)
rf_score = accuracy_score(y_test,y_pred5)
gb_score = accuracy_score(y_test,y_pred6)

final_data = pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GB'], 'ACC':[lr_score*100,svm_score*100,knn_score*100,dt_score*100,rf_score*100,gb_score*100]})
print('Models along with their Accuracy is:')
print(final_data)

print('Press Enter to view accuracy graph and save model')

sns.barplot(data=final_data, x="Models", y="ACC")
plt.show()


lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X, Y)

import joblib

# Save the model as a pickle in a file
joblib.dump(knn, 'prediction.pkl')
print('Model Saved Successfully in parent directory')