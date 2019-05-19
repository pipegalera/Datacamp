# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv("datasets/cc_approvals.data", header = None)

# Inspect data
cc_apps.head()

# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Inspect missing values in the dataset
cc_apps.tail()

# Import numpy
import numpy as np

# Inspect missing values in the dataset
print(cc_apps.tail(17))

# Replace the '?'s with NaN
cc_apps = cc_apps.replace("?", np.nan)

# Inspect the missing values again
cc_apps.tail(17)

# Impute the missing values with mean imputation
cc_apps.fillna(np.nan, inplace=True)

# Count the number of NaNs in the dataset to verify
cc_apps.isnull()

# Imput missing values with most common values
for col in cc_apps.columns:
    if cc_apps[col].dtypes == 'object':
        cc_apps[col] = cc_apps[col].fillna(cc_apps[col].value_counts().index[0])

cc_apps.isnull().values.sum()

# Label encoding

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

for col in cc_apps.columns:
    if cc_apps[col].dtypes =='object':
        cc_apps[col]=le.fit_transform(cc_apps[col])


# Features selection: drop DriversLicense and ZipCode
cc_apps = cc_apps.drop([cc_apps.columns[11],cc_apps.columns[13]], axis=1)

# Convert the DataFrame to a NumPy array
cc_apps = cc_apps.values

# Splitting the data into test and train

from sklearn.model_selection import train_test_split
X,y = cc_apps[:,0:13] , cc_apps[:,13]
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size= 0.33,
                                random_state= 42)

# Scale

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

#Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(rescaledX_train, y_train)

# Accuracy

from sklearn.metrics import confusion_matrix

y_pred = logreg.predict(rescaledX_test)
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))
print(confusion_matrix(y_test, y_pred))

# Grid search of hyperparameters
from sklearn.model_selection import GridSearchCV

tol = [0.01, 0.001, 0.0001]
max_iter = [100,150,200]

param_grid = dict(tol = tol, max_iter = max_iter)

# Chossing the best hyperparameters
grid_model = GridSearchCV(estimator= logreg , param_grid= param_grid, cv= 5)

rescaledX = scaler.fit_transform(X)
grid_model_result = grid_model.fit(rescaledX, y)

best_score, best_params = grid_model_result.best_score_,grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
