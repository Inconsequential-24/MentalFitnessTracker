#import tensorflow as tf
import pandas as pd
#import numpy as np
import seaborn as sns
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

mental_disease_data=pd.read_csv("mental-and-substance-use-as-share-of-disease -AI.csv") #PATH TO THE FILE
substance_use_data=pd.read_csv("prevalence-by-mental-and-substance-use-disorder _AI.csv") #ADD PATH TO THE FILE

df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

# Calculate the correlation matrix
corr = df.corr(numeric_only=True)
# Set up the figure size for the heatmap
plt.figure(figsize=(15, 12))
# Plot the heatmap using Seaborn
sns.heatmap(corr)
# Display the plot
plt.show()