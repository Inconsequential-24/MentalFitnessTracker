import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mental_disease_data=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv") #PATH TO THE FILE
substance_use_data=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv") #ADD PATH TO THE FILE

df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

# Scatter plot for two columns
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
                y='Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)', data=df)
plt.xlabel('Prevalence - Eating disorders')
plt.ylabel('Prevalence - Anxiety disorders')
plt.title('Scatter Plot between Prevalence - Eating disorders and Prevalence - Anxiety disorders')
plt.show()