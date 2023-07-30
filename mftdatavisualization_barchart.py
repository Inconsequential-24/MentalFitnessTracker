#import tensorflow as tf
import pandas as pd
#import numpy as np
import seaborn as sns
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mental_disease_data=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv") #PATH TO THE FILE
substance_use_data=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv") #ADD PATH TO THE FILE

df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

# Bar chart for a single column
plt.figure(figsize=(8, 6))
sns.barplot(x='Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',  # Replace 'Column_Name_Of_Interest' with the actual column name
            y='Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
            data=df)
plt.xlabel('Prevalence - Eating disorders')  # Replace 'Column_Name_Of_Interest' with the actual column name
plt.ylabel('Prevalence - Bipolar disorder')
plt.title('Bar Chart of Prevalence - Bipolar disorder by Column_Name_Of_Interest')
plt.xticks(rotation=45)
plt.show()
