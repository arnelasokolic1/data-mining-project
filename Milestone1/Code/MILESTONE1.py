#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

# Load the dataset from the given path
file_path = r"C:\Users\Korisnik\Downloads\archive\imdb_top_1000.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display the first 5 rows
print(df.head())


# In[2]:


df.head (10)


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


num_records = len(df)

print('Total numbers of records: {}'.format(num_records))

print(df.shape[0])
print(df.shape[1])


# In[6]:


# Check for missing values in the dataset
print(df.isnull().sum())

# Check for duplicate rows in the dataset
print(df.duplicated().any())


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset 
df = pd.read_csv("C:\\Users\\Korisnik\\Downloads\\archive\\imdb_top_1000.csv")

# Convert 'Runtime' column to numeric by removing ' min' and converting to float
df["Runtime"] = df["Runtime"].str.replace(" min", "").astype(float)

# Convert 'Gross' column to numeric by removing commas and converting to float
df["Gross"] = df["Gross"].str.replace(",", "").astype(float)

# Choose a numeric column for analysis
numeric_column = "IMDB_Rating"  # You can change this to 'Meta_score', 'Runtime', or 'Gross'

# Calculate statistics
mean_value = df[numeric_column].mean()
median_value = df[numeric_column].median()
mode_value = df[numeric_column].mode()[0]
midrange = (df[numeric_column].min() + df[numeric_column].max()) / 2

# Compute quartiles
q1 = df[numeric_column].quantile(0.25)
q3 = df[numeric_column].quantile(0.75)
min_value = df[numeric_column].min()
max_value = df[numeric_column].max()

# Five-number summary
five_number_summary = {
    "Minimum": min_value,
    "Q1": q1,
    "Median": median_value,
    "Q3": q3,
    "Maximum": max_value
}

# Print statistics
print("\nStatistics:")
print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")
print(f"Midrange: {midrange}")
print(f"Five-Number Summary: {five_number_summary}")

# Visualizing distribution
plt.figure(figsize=(10, 5))
sns.histplot(df[numeric_column], bins=20, kde=True, color='blue')
plt.xlabel(numeric_column)
plt.ylabel("Frequency")
plt.title(f"Distribution of {numeric_column}")
plt.show()


# In[11]:


import pandas as pd

df = pd.read_csv(r"C:\Users\Korisnik\Downloads\archive\imdb_top_1000.csv")

# Filling missing numbers with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Filling missing text with mode
df.fillna(df.mode().iloc[0], inplace=True)

print(df.isnull().sum())  


# In[12]:


print("Duplicates before:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates after:", df.duplicated().sum())


# In[13]:


# Convert 'Released_Year' to numeric, coercing errors to NaN
df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors='coerce')

# Convert to integer while keeping NaN values
df["Released_Year"] = df["Released_Year"].astype("Int64")

# Check new data types
print(df.dtypes)


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.boxplot(x=df["IMDB_Rating"])
plt.show()


# In[15]:


sns.histplot(df["IMDB_Rating"], bins=10, kde=True)
plt.show()


# In[16]:


df["Decade"] = (df["Released_Year"] // 10) * 10
df["Decade"].value_counts().sort_index().plot(kind="bar")
plt.show()


# In[17]:


print(df.isnull().sum())  # Check missing values in each column


# In[18]:


most_common_year = df["Released_Year"].mode()[0]
df["Released_Year"].fillna(most_common_year, inplace=True)


# In[19]:


df["Decade"] = (df["Released_Year"] // 10) * 10


# In[20]:


print(df.isnull().sum())  


# In[21]:


df["Gross"] = df["Gross"].replace(",", "", regex=True).astype(float)


# In[22]:


df.info()


# In[23]:


df = df[(df["Released_Year"] >= 1900) & (df["Released_Year"] <= 2025)]  # Keep valid years only


# In[24]:


print(df["Released_Year"].min(), df["Released_Year"].max())


# In[25]:


print(df[~df["Released_Year"].between(1900, 2025)])  


# In[26]:


print("Rows before filtering:", len(df))
df = df[(df["Released_Year"] >= 1900) & (df["Released_Year"] <= 2025)]
print("Rows after filtering:", len(df))

