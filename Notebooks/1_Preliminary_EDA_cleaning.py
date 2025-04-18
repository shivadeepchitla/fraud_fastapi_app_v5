import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ydata_profiling import ProfileReport
import scipy.stats as stats
from pandas.plotting import radviz, parallel_coordinates


file_path = os.path.join(os.getcwd(), "", "carclaims.csv")
csv_data = pd.read_csv(file_path)

# Display basic info
print("\n🔹 Dataset Info:")
print(csv_data.info())

# Display first 5 rows
print("\n🔹 First 5 Rows:")
print(csv_data.head())

# Check for missing values
print("\n🔹 Missing Values Count:")
print(csv_data.isnull().sum())

# Visualizing missing values
plt.figure(figsize=(10, 5))
sns.heatmap(csv_data.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# Summary statistics
print("\n🔹 Summary Statistics:")
print(csv_data.describe())

# Distribution of numerical features
num_cols = csv_data.select_dtypes(include=np.number).columns.tolist()

plt.figure(figsize=(15, 8))
for i, col in enumerate(num_cols[:6]):  # Limiting to first 6 numeric columns
    plt.subplot(2, 3, i + 1)
    sns.histplot(csv_data[col], bins=50, kde=True, color="blue")
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(csv_data.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()

# Display top correlations with 'Premium'
if "Premium" in csv_data.columns:
    print("\n🔹 Correlation with Premium:")
    print(csv_data.corr(numeric_only=True)["Premium"].sort_values(ascending=False))

profile = ProfileReport(csv_data, explorative=True)
profile.to_notebook_iframe()
print(profile)


# Define numerical columns
numerical_cols = ["WeekOfMonth", "WeekOfMonthClaimed", "Age", "PolicyNumber", 
                  "RepNumber", "Deductible", "DriverRating", "Year"]

# Define categorical columns
categorical_cols = ["MaritalStatus", "Make", "AccidentArea", "Sex", "Fault", "VehicleCategory", "PoliceReportFiled"]

# Set up subplots for QQ-Plots
fig, axes = plt.subplots(4, 2, figsize=(15, 12))

for i, col in enumerate(numerical_cols):
    row, col_idx = divmod(i, 2)
    stats.probplot(csv_data[col], dist="norm", plot=axes[row, col_idx])
    axes[row, col_idx].set_title(f"QQ-Plot of {col}")

plt.tight_layout()
plt.show()

# Set up subplots for KDE and Histograms
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(4, 2, i)
    sns.histplot(csv_data[col], kde=True, bins=30, stat="density", color="blue", alpha=0.6)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Density')
plt.tight_layout()
plt.show()

# Compute skewness and kurtosis
skew_kurtosis = {col: (csv_data[col].skew(), csv_data[col].kurtosis()) for col in numerical_cols}
print("Skewness and Kurtosis:", skew_kurtosis)

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(csv_data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Hexbin plot for Age vs. Deductible
plt.figure(figsize=(8, 6))
plt.hexbin(csv_data["Age"], csv_data["Deductible"], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(label='Counts')
plt.xlabel("Age")
plt.ylabel("Deductible")
plt.title("Hexbin Plot: Age vs. Deductible")
plt.show()

# Scatter plot of Age vs. DriverRating
plt.figure(figsize=(8, 6))
sns.scatterplot(x=csv_data['Age'], y=csv_data['DriverRating'], alpha=0.6)
plt.title("Scatter Plot: Age vs. DriverRating")
plt.xlabel("Age")
plt.ylabel("DriverRating")
plt.show()

# Pairplot to examine relationships
sns.pairplot(csv_data[numerical_cols], diag_kind='kde')
plt.show()

# Line plot for trend analysis
plt.figure(figsize=(10, 5))
sns.lineplot(x=csv_data['Year'], y=csv_data['Age'], marker='o', color='blue')
plt.title("Trend of Age over Years")
plt.xlabel("Year")
plt.ylabel("Age")
plt.grid(True)
plt.show()

# lm plot for regression analysis
sns.lmplot(x='Age', y='Deductible', data=csv_data, aspect=1.5)
plt.title("Linear Regression: Age vs. Deductible")
plt.show()

# Joint plot for Age vs. Deductible
sns.jointplot(x=csv_data['Age'], y=csv_data['Deductible'], kind='reg')
plt.show()

# Seasonal decomposition (if applicable)
import statsmodels.api as sm
if 'Year' in csv_data.columns and 'Age' in csv_data.columns:
    csv_data = csv_data.sort_values(by='Year')
    decomposition = sm.tsa.seasonal_decompose(csv_data['Age'], model='additive', period=1)
    decomposition.plot()
    plt.show()

# Categorical column analysis
plt.figure(figsize=(15, 10))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=csv_data[col], order=csv_data[col].value_counts().index)
    plt.title(f'Count Plot: {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Box plot for numerical vs categorical analysis
plt.figure(figsize=(12, 6))
sns.boxplot(x='VehicleCategory', y='Age', data=csv_data)
plt.title("Box Plot: Age vs. Vehicle Category")
plt.show()

# Swarm plot for numerical vs categorical analysis
plt.figure(figsize=(12, 6))
sns.swarmplot(x='VehicleCategory', y='Deductible', data=csv_data)
plt.title("Swarm Plot: Deductible vs. Vehicle Category")
plt.show()

# Bar plot for categorical analysis
plt.figure(figsize=(10, 6))
sns.barplot(x='Fault', y='Age', data=csv_data, estimator=np.mean)
plt.title("Bar Plot: Average Age by Fault")
plt.show()

# Cat plot for multiple categories
sns.catplot(x='AccidentArea', y='Age', hue='Sex', kind='bar', data=csv_data, aspect=1.5)
plt.title("Cat Plot: Age by Accident Area and Sex")
plt.show()

# Multivariate Plots
plt.figure(figsize=(10, 6))
radviz(csv_data, class_column='VehicleCategory')
plt.title("RadViz Plot: Vehicle Category")
plt.show()

plt.figure(figsize=(12, 6))
parallel_coordinates(csv_data, class_column='VehicleCategory', colormap=plt.get_cmap("Set1"))
plt.title("Parallel Coordinates Plot: Vehicle Category")
plt.xticks(rotation=45)
plt.show()
