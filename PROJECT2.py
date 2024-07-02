#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = 'EV-Dashboard-Data-as-on-20240202.csv'
data = pd.read_csv(file_path)
data


# In[2]:


# Filter data for the year 2024
data_2024 = data[data['Year'] == 2024]
data_2024


# In[3]:


# 1. Segmenting Data by Vehicle Type and Category
vehicle_type_summary_2024 = data_2024.groupby('Vehicle Type')['Total'].sum().reset_index()
vehicle_category_summary_2024 = data_2024.groupby('Vehicle Category')['Total'].sum().reset_index()

# Displaying the summaries
print("Vehicle Type Summary for 2024:")
print(vehicle_type_summary_2024)

print("\nVehicle Category Summary for 2024:")
print(vehicle_category_summary_2024)


# In[4]:


# Vehicle Type Summary Plot
vehicle_type_summary_2024['Vehicle Type'] = pd.Categorical(vehicle_type_summary_2024['Vehicle Type'])

plt.figure(figsize=(10, 6))
sns.barplot(data=vehicle_type_summary_2024, x='Total', y='Vehicle Type', palette='viridis', hue='Vehicle Type', dodge=False, legend=False)
plt.title('Total Vehicles by Vehicle Type for 2024')
plt.xlabel('Total Vehicles')
plt.ylabel('Vehicle Type')
plt.show()


# In[5]:


# Vehicle Category Summary Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=vehicle_category_summary_2024, x='Total', y='Vehicle Category', palette='viridis')
plt.title('Total Vehicles by Vehicle Category for 2024')
plt.xlabel('Total Vehicles')
plt.ylabel('Vehicle Category')
plt.show()


# In[6]:


data_2023_2024 = data[(data['Year'] == 2023) | (data['Year'] == 2024)]

# 2. Analyzing EV Trends Over Time (Monthly for 2023 and 2024)
month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
monthly_trends_2023_2024 = data_2023_2024.groupby(['Year', 'Month_name'])['Total'].sum().reset_index()
monthly_trends_2023_2024['Month_name'] = pd.Categorical(monthly_trends_2023_2024['Month_name'], categories=month_order, ordered=True)
monthly_trends_2023_2024 = monthly_trends_2023_2024.sort_values(by=['Year', 'Month_name']).reset_index(drop=True)

# Displaying the monthly trends for 2023 and 2024
print("\nMonthly Trends for 2023 and 2024:")
print(monthly_trends_2023_2024)


# In[7]:


# Monthly Trends Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_trends_2023_2024, x='Month_name', y='Total', hue='Year', marker='o')
plt.title('Monthly EV Trends for 2023 and 2024')
plt.xlabel('Month')
plt.ylabel('Total Vehicles')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.show()


# In[8]:


# 3. Comparing EV Numbers Across Different States for 2024
state_summary_2024 = data_2024.groupby('State')['Total'].sum().reset_index().sort_values(by='Total', ascending=False)

# Displaying the state summary for 2024
print("\nState Summary for 2024:")
print(state_summary_2024)


# In[9]:


# State Summary Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=state_summary_2024, x='Total', y='State', palette='viridis')
plt.title('Total Vehicles by State for 2024')
plt.xlabel('Total Vehicles')
plt.ylabel('State')
plt.show()


# In[10]:


# 5. Classification Example: K-Means Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Prepare data for clustering
clustering_data = state_summary_2024[['State', 'Total']]
clustering_data['Total'] = StandardScaler().fit_transform(clustering_data[['Total']])

# Applying K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(clustering_data[['Total']])

# Displaying cluster results
print("\nClustering Results:")
print(clustering_data)

# Visualization of Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=clustering_data, x='State', y='Total', hue='Cluster', palette='viridis', s=100)
plt.title('State Clusters Based on Total Vehicles for 2024')
plt.xlabel('State')
plt.xticks(rotation=90) 
plt.ylabel('Standardized Total Vehicles')
plt.legend(title='Cluster')
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the CSV file
file_path = 'EV-Dashboard-Data-as-on-20240202.csv'
data = pd.read_csv(file_path)

# Filter data for the year 2024
data_2024 = data[data['Year'] == 2024]

# Summarize data by state
state_summary_2024 = data_2024.groupby('State')['Total'].sum().reset_index()

# Displaying the state summary for 2024
print("\nState Summary for 2024:")
print(state_summary_2024)

# Prepare data for clustering
clustering_data_2024 = state_summary_2024[['State', 'Total']]

# Hierarchical Clustering for States
linked = linkage(clustering_data_2024['Total'].values.reshape(-1, 1), method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linked, labels=clustering_data_2024['State'].values, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram for States (2024)')
plt.xlabel('State')
plt.ylabel('Distance')
plt.xticks(rotation=90)
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Bar Plot for State EV Counts in 2024
plt.figure(figsize=(12, 8))
sns.barplot(data=state_summary_2024, x='Total', y='State', palette='viridis')
plt.title('Total EVs by State in 2024')
plt.xlabel('Total EVs')
plt.ylabel('State')
plt.xticks(rotation=90)
plt.show()


# In[13]:


pip install --upgrade numpy pandas scipy scikit-learn matplotlib seaborn


# In[14]:


pip install numpy pandas scipy scikit-learn matplotlib seaborn


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the CSV file
file_path = 'EV-Dashboard-Data-as-on-20240202.csv'
data = pd.read_csv(file_path)

# Filter data for the years 2023 and 2024
data_2023_2024 = data[(data['Year'] == 2023) | (data['Year'] == 2024)]

# Monthly Trends Data Preparation
monthly_trends = data_2023_2024.groupby(['Year', 'Month_name'])['Total'].sum().reset_index()
monthly_trends['Month'] = monthly_trends.apply(lambda row: f"{row['Year']}-{row['Month_name']}", axis=1)
monthly_trends['Month'] = pd.to_datetime(monthly_trends['Month'], format='%Y-%b')
monthly_trends = monthly_trends.sort_values(by='Month')
monthly_trends['Month_numeric'] = monthly_trends['Month'].dt.month + (monthly_trends['Year'] - 2023) * 12

# Displaying the monthly trends for 2023 and 2024
print("\nMonthly Trends for 2023 and 2024:")
print(monthly_trends)

X = monthly_trends[['Month_numeric']]
y = monthly_trends['Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Predictive Modeling of EV Trends for 2023 and 2024')
plt.xlabel('Month Numeric')
plt.ylabel('Total Vehicles')
plt.legend()
plt.show()

# Visualize monthly trends with the regression line
plt.figure(figsize=(10, 6))
sns.lineplot(data=monthly_trends, x='Month', y='Total', hue='Year', marker='o')
plt.plot(monthly_trends['Month'], model.predict(monthly_trends[['Month_numeric']]), color='red', linestyle='--', label='Prediction')
plt.title('Monthly EV Trends for 2023 and 2024 with Predictive Model')
plt.xlabel('Month')
plt.ylabel('Total Vehicles')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[16]:


# Calculate growth rate of EV adoption for each state
state_growth_rate = data_2024.groupby('State')['Total'].sum().reset_index()
state_growth_rate['Growth Rate'] = state_growth_rate['Total'].pct_change().fillna(0)
state_growth_rate['Growth Rate']


# In[17]:


# Plot growth rate by state
plt.figure(figsize=(14, 8))
sns.barplot(data=state_growth_rate, x='Growth Rate', y='State', hue='State', palette='coolwarm', dodge=False, legend=False)
plt.title('EV Adoption Growth Rate by State for 2024')
plt.xlabel('Growth Rate')
plt.ylabel('State')
plt.show()


# In[20]:


from sklearn.cluster import KMeans
cluster_data = state_summary_2024[['Total']]


kmeans = KMeans(n_clusters=3, random_state=42)
state_summary_2024['Cluster'] = kmeans.fit_predict(cluster_data)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=state_summary_2024, x='State', y='Total', hue='Cluster', palette='viridis')
plt.title('KMeans Clustering of States by Total EVs for 2024')
plt.xlabel('State')
plt.ylabel('Total EVs')
plt.xticks(rotation=90)
plt.show()

