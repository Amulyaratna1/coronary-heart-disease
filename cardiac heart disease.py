#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the dataframe using pandas
df = pd.read_csv(r"C:\Users\Dell\Downloads\cardio disease.csv")


# In[3]:


#displaying the dataframe
df1 = df


# In[4]:


df1


# In[5]:


#displaying first five rows of dataframe
df1.head()


# In[6]:


##displaying last five rows of dataframe
df1.tail()


# In[7]:


df1.info()


# In[8]:


df1.describe()


# In[9]:


#displaying the column names
df1.columns


# In[10]:


#displaying the number of rows and column
df1.shape


# In[11]:


#dropping the first column, because of unique values in that column
df1 = df1.drop(columns='id')


# In[12]:


df1 = df1.drop(columns='education')


# In[13]:


df1.head()


# In[14]:


#checking the shape of dataframe after dropping the unecessary column
df1.shape


# In[15]:


df1.isnull().sum()


# In[16]:


df1.isnull().sum().sum()


# In[17]:


#checking the special characters in the data frame 
column_name = 'BPMeds'

has_special_characters = df1[df1[column_name].notnull() & df[column_name].astype(str).str.contains(r'[^a-zA-Z0-9\s]')]

print(has_special_characters[column_name])


# In[18]:


df1.duplicated().sum()


# In[19]:


sns.boxplot(data=df1)


# In[20]:


sns.boxplot(df1['cigsPerDay'])
plt.show()


# In[21]:


sns.boxplot(df1['totChol'])
plt.show


# In[22]:


sns.boxplot(df1['sysBP'])
plt.show


# In[23]:


sns.boxplot(df1['diaBP'])
plt.show


# In[24]:


sns.boxplot(df1['BMI'])
plt.show


# In[25]:


sns.boxplot(df1['heartRate'])
plt.show


# In[26]:


sns.boxplot(df1['glucose'])
plt.show


# In[27]:


#displaying 10 sample
df1.sample(10)


# In[28]:


df1.sample(10)


# In[29]:


# Drop rows with null values in the 'Column1' and 'Column2' columns
df1 = df1.dropna(subset = ['heartRate','BPMeds'])


# In[30]:


# If you want to fill only specific columns, you can specify the subset
# For example, to fill missing values in 'Column1' and 'Column2' with their respective means:
df1[['cigsPerDay', 'totChol']] = df1[['cigsPerDay', 'totChol',]].fillna(df1.mean())


# In[31]:


# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Fill missing values in 'specific_column' with the median of that column
median_value = df1['BMI'].median()
df1['BMI'].fillna(median_value, inplace=True)


# In[32]:


# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Fill missing values in 'specific_column' with the median of that column
median_value = df1['glucose'].median()
df1['glucose'].fillna(median_value, inplace=True)


# In[33]:


df1.isnull().sum()


# In[34]:


sns.boxplot(data=df1)


# In[35]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Replace 'specific_column' with the name of the column you want to analyze for outliers

# Calculate the first quartile (Q1)
Q1 = df1['totChol'].quantile(0.25)
Q1


# In[36]:


# Calculate the third quartile (Q3)
Q3 = df1['totChol'].quantile(0.75)
Q3


# In[37]:


# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1
IQR


# In[38]:


# Calculate the lower bound and upper bound for outlier detection
lower_bound = Q1 - 1.5 * IQR
lower_bound


# In[39]:


upper_bound = Q3 + 1.5 * IQR
upper_bound


# In[40]:


# Find the outliers
outliers = df1[(df1['totChol'] < lower_bound) | (df1['totChol'] > upper_bound)]
outliers


# In[41]:


median_value = df1['totChol'].median()

# Replace outliers with the chosen treatment
df1['totChol'] = df1['totChol'].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
    # Replace 'mean_value' with 'median_value' or 'mode_value' for different treatments
)


# In[42]:


sns.boxplot(df1['totChol'])


# In[43]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Replace 'specific_column' with the name of the column you want to analyze for outliers

# Calculate the first quartile (Q1)
Q1 = df1['sysBP'].quantile(0.25)
Q1


# In[44]:


# Calculate the third quartile (Q3)
Q3 = df1['sysBP'].quantile(0.75)
Q3


# In[45]:


# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1
IQR


# In[46]:


# Calculate the lower bound and upper bound for outlier detection
lower_bound = Q1 - 1.5 * IQR
lower_bound


# In[47]:


upper_bound = Q3 + 1.5 * IQR
upper_bound


# In[48]:


# Find the outliers
outliers = df1[(df1['sysBP'] < lower_bound) | (df1['sysBP'] > upper_bound)]
outliers


# In[49]:


median_value = df1['sysBP'].median()

# Replace outliers with the chosen treatment
df1['sysBP'] = df1['sysBP'].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
    # Replace 'mean_value' with 'median_value' or 'mode_value' for different treatments
)


# In[50]:


sns.boxplot(df1['sysBP'])


# In[51]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Replace 'specific_column' with the name of the column you want to analyze for outliers

# Calculate the first quartile (Q1)
Q1 = df1['diaBP'].quantile(0.25)
Q1


# In[52]:


# Calculate the third quartile (Q3)
Q3 = df1['diaBP'].quantile(0.75)
Q3


# In[53]:


# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1
IQR


# In[54]:


# Calculate the lower bound and upper bound for outlier detection
lower_bound = Q1 - 1.5 * IQR
lower_bound


# In[55]:


upper_bound = Q3 + 1.5 * IQR
upper_bound


# In[56]:


# Find the outliers
outliers = df1[(df1['diaBP'] < lower_bound) | (df1['diaBP'] > upper_bound)]
outliers


# In[57]:


median_value = df1['diaBP'].median()

# Replace outliers with the chosen treatment
df1['diaBP'] = df1['diaBP'].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
    # Replace 'mean_value' with 'median_value' or 'mode_value' for different treatments
)


# In[58]:


sns.boxplot(df1['diaBP'])


# In[59]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Replace 'specific_column' with the name of the column you want to analyze for outliers

# Calculate the first quartile (Q1)
Q1 = df1['BMI'].quantile(0.25)
Q1


# In[60]:


Q3 = df1['BMI'].quantile(0.75)
Q3


# In[61]:


# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1
IQR


# In[62]:


# Calculate the lower bound and upper bound for outlier detection
lower_bound = Q1 - 1.5 * IQR
lower_bound


# In[63]:


# Calculate the upper bound and upper bound for outlier detection
upower_bound = Q1 + 1.5 * IQR
upower_bound


# In[64]:


# Find the outliers
outliers = df1[(df1['BMI'] < lower_bound) | (df1['BMI'] > upper_bound)]
outliers


# In[65]:


median_value = df1['BMI'].median()

# Replace outliers with the chosen treatment
df1['BMI'] = df1['BMI'].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
    # Replace 'mean_value' with 'median_value' or 'mode_value' for different treatments
)


# In[66]:


sns.boxplot(df1['BMI'])


# In[67]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Replace 'specific_column' with the name of the column you want to analyze for outliers

# Calculate the first quartile (Q1)
Q1 = df1['heartRate'].quantile(0.25)
Q1


# In[68]:


Q3 = df1['heartRate'].quantile(0.75)
Q3


# In[69]:


# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1
IQR


# In[70]:


# Calculate the lower bound and upper bound for outlier detection
lower_bound = Q1 - 1.5 * IQR
lower_bound


# In[71]:


# Calculate the upper bound and upper bound for outlier detection
upower_bound = Q1 + 1.5 * IQR
upower_bound


# In[72]:


# Find the outliers
outliers = df1[(df1['heartRate'] < lower_bound) | (df1['heartRate'] > upper_bound)]
outliers


# In[73]:


edian_value = df1['heartRate'].median()

# Replace outliers with the chosen treatment
df1['heartRate'] = df1['heartRate'].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
    # Replace 'mean_value' with 'median_value' or 'mode_value' for different treatments
)


# In[74]:


sns.boxplot(df1['heartRate'])


# In[75]:


import pandas as pd

# Assuming you have a DataFrame called 'df' with a column 'specific_column'
# Replace 'specific_column' with the name of the column you want to analyze for outliers

# Calculate the first quartile (Q1)
Q1 = df1['glucose'].quantile(0.25)
Q1


# In[76]:


Q3 = df1['glucose'].quantile(0.75)
Q3


# In[77]:


# Calculate the IQR (Interquartile Range)
IQR = Q3 - Q1
IQR


# In[78]:


# Calculate the lower bound and upper bound for outlier detection
lower_bound = Q1 - 1.5 * IQR
lower_bound


# In[79]:


# Calculate the upper bound and upper bound for outlier detection
upower_bound = Q1 + 1.5 * IQR
upower_bound


# In[80]:


# Find the outliers
outliers = df1[(df1['glucose'] < lower_bound) | (df1['glucose'] > upper_bound)]
outliers


# In[81]:


edian_value = df1['glucose'].median()

# Replace outliers with the chosen treatment
df1['glucose'] = df1['glucose'].apply(
    lambda x: median_value if x < lower_bound or x > upper_bound else x
    # Replace 'mean_value' with 'median_value' or 'mode_value' for different treatments
)


# In[82]:


sns.boxplot(df1['glucose'])


# In[83]:


import pandas as pd
df1=pd.get_dummies(df1,columns=['sex','is_smoking'],drop_first=True)


# In[84]:


df1


# In[85]:


df1.shape


# In[86]:


# Calculate the correlation matrix
correlation_matrix = df1.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[87]:


df1 = df1.drop(columns='diaBP')


# In[88]:


df1


# In[104]:


# Assuming 'dependent_var' is the name of your dependent variable and 'df' is your DataFrame
correlation_matrix = df1.corr()
correlation_with_dependent = correlation_matrix['TenYearCHD'].drop('TenYearCHD')
print(correlation_with_dependent)


# In[110]:


df1.drop(['heartRate' , 'cigsPerDay' , 'BPMeds' ,'prevalentStroke' , 'BMI'] ,axis=1 , inplace=True)


# In[111]:


for i, column in enumerate(columns_to_visualize, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[112]:


# Assuming 'target' is the name of the target attribute
x = df1.drop('TenYearCHD', axis=1)
y = df1['TenYearCHD']


# In[113]:


x.shape


# In[114]:


y


# In[115]:


from sklearn.datasets import make_classification
x, y = make_classification(n_samples=3000, n_classes=2, weights=[1,1], random_state=42 , n_features=13)


# In[116]:


from sklearn.preprocessing import StandardScaler


# In[117]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
X_test_scaled = scaler.transform(X_test)


# In[118]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
# Create and train the logistic regression model
model_2 = LogisticRegression()
model_2.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model_2.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)


print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)
print("Recall:", recall)
print("Precision:", precision)


# In[120]:


import pickle

#Open a file in binary write mode
file_path = 'logisticregression.pkl'

with open(file_path, 'wb') as file:

 pickle.dump(model_2, file)


# In[121]:


pwd


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




