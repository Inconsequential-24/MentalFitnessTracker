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

# Function to provide personalized message based on the predicted score
def get_mental_health_message(score):
    if score < 3:
        return "Your mental fitness score is low. We recommend seeking professional help. Here is a helpline link: https://indianhelpline.com/suicide-helpline or http://healthcollective.in/contact/helplines/"
    elif 3 <= score < 5:
        return "Your mental fitness score is between 3 to 5. Please pay attention to your mental health and consider seeking support if needed."
    elif 5 <= score < 7:
        return "Your mental fitness score is above average. Keep working on your mental health to improve further."
    elif 7 <= score < 8:
        return "Congratulations! Your mental fitness score is between 7 to 8. You are doing well. Consider incorporating regular exercise into your routine for even better mental health."
    elif 8 <= score <= 10:
        return "Congratulations! Your mental fitness score is between 8 to 10. You have a good mental health score. Keep up with your healthy lifestyle. However, don't hesistate to ask for help whenever required."

# Get input from the user for the features used in the model
input_features = [
    float(input("Enter the prevalence of Schizophrenia (Percent): ")),
    float(input("Enter the prevalence of Bipolar disorder (Percent): ")),
    float(input("Enter the prevalence of Eating disorders (Percent): ")),
    float(input("Enter the prevalence of Anxiety disorders (Percent): ")),
    float(input("Enter the prevalence of Drug use disorders (Percent): ")),
    float(input("Enter the prevalence of Depressive disorders (Percent): "))
]

# Convert the input features into a 2D array (required for prediction)
input_data = np.array([input_features])

# Make predictions using the trained model
predicted_mental_fitness = model.predict(input_data)

# Display the predicted mental fitness score
print("Predicted Mental Fitness Score:", predicted_mental_fitness[0])

# Get the personalized message based on the predicted score
message = get_mental_health_message(predicted_mental_fitness[0])
print(message)
