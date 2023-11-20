# Importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset into dataframe

my_data = pd.read_csv("placement-data.csv")

print(f'Number of rows:{my_data.shape[0]}')
print(f'Number of columns:{my_data.shape[1]}')

# Convert Objects into Categories

my_data['PlacementStatus'] = my_data['PlacementStatus'].astype('category')
my_data['ExtracurricularActivities'] = my_data['ExtracurricularActivities'].astype('category')
my_data['PlacementTraining'] = my_data['PlacementTraining'].astype('category')

# Convert Categories char into 0 and 1

my_data['PlacementStatus']=my_data['PlacementStatus'].map({'Placed':1,'NotPlaced':0})
my_data['ExtracurricularActivities']=my_data['ExtracurricularActivities'].map({'Yes':1,'No':0})
my_data['PlacementTraining']=my_data['PlacementTraining'].map({'Yes':1,'No':0})

# Feature Selection

X = my_data.drop(columns=['StudentID','PlacementStatus','SSC_Marks'],axis=1)
Y = my_data['PlacementStatus']

# Fit the whole dataset for training

lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X, Y)

# Saving the MODEL

joblib.dump(lr, 'prediction.pkl')
print('Model Saved Successfully in parent directory')