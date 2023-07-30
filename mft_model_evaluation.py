import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

# Create and train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train.ravel()) 

# Fit the Random Forest Regressor model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2) Score
r2 = r2_score(y_test, y_pred)

# Calculate success rate and failure rate
threshold = 1  # Adjust the threshold as per your requirements
success_rate = np.sum(np.abs(y_test - y_pred) <= threshold) / len(y_test) * 100
failure_rate = 100 - success_rate

# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Mental Fitness Score')
plt.ylabel('Predicted Mental Fitness Score')
plt.title('Scatter Plot for Random Forest Regressor')
plt.show()

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
print("Success Rate:", success_rate, "%")
print("Failure Rate:", failure_rate, "%")
