import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

mental_disease_data=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv") #PATH TO THE FILE
substance_use_data=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv") #ADD PATH TO THE FILE

#Conactenate the two data frames
df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

#Data Preprocessing
#df.drop(['Entity','Code','Year'],axis=1,inplace=True)
df=df.fillna(df.mean(numeric_only=True))

x=df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
       'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',]].to_numpy()

y=df[['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']].to_numpy()
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train.ravel())  # Convert y_train to a 1D array using ravel()

# Make predictions on the test data
y_pred = model.predict(x_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared (Coefficient of Determination) to measure the model's goodness of fit
r2 = r2_score(y_test, y_pred)

# Define success and failure thresholds (You can set these based on your specific requirements)
success_threshold = 0.2
failure_threshold = 0.4

# Calculate the success rate as a percentage
success_rate = sum(abs(y_pred - y_test.ravel()) < success_threshold) / len(y_test) * 100

# Calculate the failure rate as a percentage
failure_rate = sum(abs(y_pred - y_test.ravel()) > failure_threshold) / len(y_test) * 100

# Print the metrics
print("Mean Squared Error (MSE)for Decision Tree Regressor:", mse)
print("R-squared (R2):", r2)
print("Success Rate:", success_rate, "%")
print("Failure Rate:", failure_rate, "%")
