# Importing necessary libraries
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Reading data from CSV into a pandas DataFrame
df = pd.read_csv('Testdata.csv',low_memory=False)
print(df.head(5))

# Computing summary statistics of the DataFrame
print(df.describe())
print(df.isnull().sum())

# Generating a data profiling report and saving it as an HTML file
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file('EDA_Tradedata_Analysis.html')

# Dropping rows with missing values
print(df.dropna(inplace=True))

# Selecting numerical columns for scaling
numerical_columns = ['tradeId','version', 'rate', 'price','PartyId']

# Standardizing 
scaler = StandardScaler()
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
data_standardized = scaler.fit_transform(df[numerical_columns])

# Normalizing 
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df[numerical_columns])
data_normalized_df = pd.DataFrame(data_normalized, columns=numerical_columns)
print(data_normalized_df)

#correlation matrix
correlation_matrix = df[numerical_columns].corr()
print(correlation_matrix)

# Encoding categorical features using one-hot encoding
categorical_features = ['regulator', 'assetClass', 'clStatus', 'cflag', 'eFlag', 'method', 'eventT', 'mType', 'sType', 'transactionType', 'Reporting Status']
encoded_data = pd.get_dummies(df, columns=categorical_features)

# Grouping data by 'transactionType' and counting occurrences
grouped_data = df.groupby('transactionType').size().reset_index(name='Counts')
print(grouped_data)