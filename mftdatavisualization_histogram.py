import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


mental_disease_data=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv") #PATH TO THE FILE
substance_use_data=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv") #ADD PATH TO THE FILE

df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

# Histogram for a single column
plt.figure(figsize=(8, 6))
sns.histplot(df['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)'], kde=True)
plt.xlabel('Prevalence - Schizophrenia')
plt.ylabel('Frequency')
plt.title('Histogram of Prevalence - Schizophrenia')
plt.show()