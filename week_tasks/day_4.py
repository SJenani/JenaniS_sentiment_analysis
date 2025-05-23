# -*- coding: utf-8 -*-
"""day_4

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZNUysau7RdcjhLqGeHIYlW8_nbzSyB2q
"""

import  pandas as pd
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, None, 22, 28],  # Notice the missing value (None) in Age for Charlie
    'City': ['New York', 'Los Angeles', 'Chicago', None, 'Miami']  # Missing value in City for David
}
df=pd.DataFrame(data)
df_cleaned = df.dropna()  # Removes rows with missing values
print(df_cleaned)

import pandas as pd
# Creating a sample DataFrame with missing values (NaN)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, None, 22, 28],  # Notice the missing value (None) in Age for Charlie
    'City': ['New York', 'Los Angeles', 'Chicago', None, 'Miami'],  # Missing value in City for David
    'Marks': [80, 90, 75, None, 85], # Adding a 'Marks' column with a missing value
    'Attendance': [95, 92, 88, None, 90]  # Adding an 'Attendance' column with a missing value
}
df = pd.DataFrame(data)
df_cleaned = df.dropna()  # Removes rows with missing values
print(df_cleaned)
# Filling missing values
df["Age"].fillna(df["Age"].mean(), inplace=True)
# Only fill if the 'Marks' column exists
if 'Marks' in df.columns:
    df["Marks"].fillna(df["Marks"].median(), inplace=True)
# Only fill if the 'Attendance' column exists
if 'Attendance' in df.columns:
    df["Attendance"].fillna(df["Attendance"].mean(), inplace=True)
print(df)

import pandas as pd

# Creating a sample DataFrame with missing values (NaN)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, None, 22, 28],  # Notice the missing value (None) in Age for Charlie
    'City': ['New York', 'Los Angeles', 'Chicago', None, 'Miami'],  # Missing value in City for David
    'Marks': [80, 90, 75, None, 85], # Adding a 'Marks' column with a missing value
    'Attendance': [95, 92, 88, None, 90]  # Adding an 'Attendance' column with a missing value
}

df = pd.DataFrame(data)

# Drop rows with missing values
df_cleaned = df.dropna()  # Removes rows with missing values
print("DataFrame after dropping rows with missing values:")
print(df_cleaned)

# Filling missing values
df["Age"]=df["Age"].fillna(df["Age"].mean(), inplace=True)  # Fill 'Age' with the mean of 'Age'

# Check if 'Marks' column exists before filling missing values
if 'Marks' in df.columns:
    df["Marks"]=df["Marks"].fillna(df["Marks"].median(), inplace=True)  # Fill 'Marks' with the median of 'Marks'

# Check if 'Attendance' column exists before filling missing values
if 'Attendance' in df.columns:
    df["Attendance"]=df["Attendance"].fillna(df["Attendance"].mean(), inplace=True)  # Fill 'Attendance' with the mean of 'Attendance'

# Print the DataFrame after filling missing values
print("\nDataFrame after filling missing values:")
print(df)

import pandas as pd

# Creating a sample DataFrame with missing values (NaN)
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, None, 22, 28],  # Notice the missing value (None) in Age for Charlie
    'City': ['New York', 'Los Angeles', 'Chicago', None, 'Miami'],  # Missing value in City for David
    'Marks': [80, 90, 75, None, 85], # Adding a 'Marks' column with a missing value
    'Attendance': [95, 92, 88, None, 90]  # Adding an 'Attendance' column with a missing value
}

df = pd.DataFrame(data)

# Drop rows with missing values
df_cleaned = df.dropna()  # Removes rows with missing values
print("DataFrame after dropping rows with missing values:")
print(df_cleaned)

# Filling missing values
df["Age"]=df["Age"].fillna(df["Age"].mean(), inplace=True)  # Fill 'Age' with the mean of 'Age'

# Check if 'Marks' column exists before filling missing values
if 'Marks' in df.columns:
    df["Marks"]=df["Marks"].fillna(df["Marks"].median(), inplace=True)  # Fill 'Marks' with the median of 'Marks'

# Check if 'Attendance' column exists before filling missing values
if 'Attendance' in df.columns:
    df["Attendance"]=df["Attendance"].fillna(df["Attendance"].mean(), inplace=True)  # Fill 'Attendance' with the mean of 'Attendance'

# Print the DataFrame after filling missing values
print("\nDataFrame after filling missing values:")
print(df)

import pandas as pd

# Sample data with some missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, None, 22, 28],  # None represents a missing value for Charlie
    'City': ['New York', 'Los Angeles', 'Chicago', None, 'Miami']  # None represents a missing value for David
}

# Create DataFrame from the data dictionary
df = pd.DataFrame(data)

# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Check for missing values (NaN) in each column and print the count
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Optionally, handle missing values by filling them with a default value or dropping rows
# Example: Filling missing 'Age' with the average age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Example: Dropping rows where any column has a missing value
# df_cleaned = df.dropna()

# Print the DataFrame after handling missing values
print("\nDataFrame After Handling Missing Values:")
print(df)

import pandas as pd

# Create the DataFrame with 5 rows
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [23, 24, 22, None, 21],  # Age values
    'Marks': [79, 97, 61, 51, None],  # Marks values (out of 100)
    'Attendance': [80, None, 93, 79, 88],
    'Passed': [True,True,False,True,None]# Attendance values (percentage)
}

# Create the DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
print(df.isnull().sum())
df_cleaned = df.dropna()  # Removes rows with missing values
print(df_cleaned)

import pandas as pd

# Create the DataFrame with 5 rows
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [23, 24, 22, None, 21],  # Age values
    'Marks': [79, 97, 61, 51, None],  # Marks values (out of 100)
    'Attendance': [80, None, 93, 79, 88],
    'Passed': [True,True,False,True,None]# Attendance values (percentage)
}

# Create the DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
print("\n")
print(df.isnull().sum())
print("\n")
df_cleaned = df.dropna()  # Removes rows with missing values
print(df_cleaned)
print("\n")
df.fillna({"Age":df["Age"].mean(),"Marks":df["Marks"].median(),"Attendance":df["Attendance"].mean()}, inplace=True)
df=df.fillna({"Age":df["Age"].mean(),"Marks":df["Marks"].median(),"Attendance":df["Attendance"].mean()})
print(df)
print("\n")
df.fillna({"Passed":df["Passed"].mode()},inplace=True)
df=df.fillna({"Passed":df["Passed"].mode()})
print(df)
print("\n")
df.ffill(inplace=True)  # Forward fill
df.bfill(inplace=True)
print(df)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[["Marks", "Attendance"]] = scaler.fit_transform(df[["Marks", "Attendance"]])
print(df_scaled)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled[["Marks", "Attendance"]] = scaler.fit_transform(df[["Marks", "Attendance"]])
print(df_scaled)
df_encoded = pd.get_dummies(df, columns=["Passed"],
drop_first=True)
print(df_encoded)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df["Passed"] = encoder.fit_transform(df["Passed"])
print(df)

def performance_category(marks):
    if marks >= 85:
        return "High"
    elif marks >= 70:
        return "Medium"
    else:
        return "Low"

df["Performance"] = df["Marks"].apply(performance_category)
print(df)
df["Age_Group"] = pd.cut(df["Age"], bins=[18, 21, 24],
labels=["Young", "Adult"])
print(df)