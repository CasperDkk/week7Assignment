import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('iris.csv', header=None)
# Assign column names
df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']

# Display the first few rows
print(df.head())

# Get summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Group by Species and calculate mean
species_means = df.groupby('Species').mean()
print(species_means)

# Scatter plot for SepalLength vs SepalWidth
plt.scatter(df['SepalLength'], df['SepalWidth'], c='blue', label='Sepal')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Width')
plt.legend()
plt.show()

# Histogram for PetalLength
plt.hist(df['PetalLength'], bins=20, color='green', edgecolor='black')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.title('Histogram of Petal Length')
plt.show()

# Box plot for Sepal Length grouped by Species
df.boxplot(column='SepalLength', by='Species')
plt.title('Sepal Length by Species')
plt.suptitle('')  # Suppress the automatic suptitle
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()


