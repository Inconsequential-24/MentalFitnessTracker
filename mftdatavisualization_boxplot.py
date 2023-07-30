import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mental_disease_data=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv") #PATH TO THE FILE
substance_use_data=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv") #ADD PATH TO THE FILE

df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

# Box plot for multiple columns
selected_columns = ['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
                    'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)']

plt.figure(figsize=(8, 6))
sns.boxplot(data=df[selected_columns])
plt.ylabel('Prevalence')
plt.title('Box Plot of Prevalence - Drug use disorders and Prevalence - Depressive disorders')
plt.xticks(rotation=45)
plt.show()