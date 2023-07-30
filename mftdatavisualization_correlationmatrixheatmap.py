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

df=pd.concat(objs=[substance_use_data,mental_disease_data],axis=1)

# Plotting the correlation matrix heatmap
plt.figure(figsize=(10, 8))
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pair plot for visualizing relationships between selected columns
selected_columns = ['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
                    'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
                    'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
                    'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
                    'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
                    'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)']

sns.pairplot(df[selected_columns])
plt.show()