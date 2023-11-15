import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import csv

# Load the data from the CSV file
with open(r'/run/media/arash/New Volume/uniBowlshit/DataMining/Projects/project1 datamining/insurance.csv', "r") as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    data = list(csvreader)
    
# Extract the data for each attribute
ages = [float(row[0]) for row in data]
bmis = [float(row[2]) for row in data]
children = [int(row[3]) for row in data]
charges = [float(row[6]) for row in data]

# Define a function to compute statistics for a list of values
def compute_statistics(values):
    max_value = max(values)
    min_value = min(values)
    q1 = np.percentile(values, 25)
    median = np.median(values)
    q3 = np.percentile(values, 75)
    std_dev = np.std(values)
    mean = np.mean(values)
    return max_value, min_value, q1, q3, median, std_dev, mean

# Compute statistics for each attribute
age_stats = compute_statistics(ages)
bmi_stats = compute_statistics(bmis)
children_stats = compute_statistics(children)
charges_stats = compute_statistics(charges)

# Print the statistics for each attribute
print("Age Statistics:")
print("Maximum:", age_stats[0])
print("Minimum:", age_stats[1])
print("First Quartile (Q1):", age_stats[2])
print("Third Quartile (Q3):", age_stats[3])
print("Median:", age_stats[4])
print("Standard Deviation:", age_stats[5])
print("Mean:", age_stats[6])

# Create histograms for each attribute
plt.figure(figsize=(12, 8))

# Age Histogram
plt.subplot(221)
plt.hist(ages, bins=20, edgecolor='k', alpha=0.75)
plt.title('Age Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Scatterplots
scatter_axes = plt.figure(figsize=(16, 4))

# Age vs. Charges Scatterplot
scatter_axes.add_subplot(141)
plt.scatter(ages, charges, alpha=0.5)
plt.title('Age vs. Charges')
plt.xlabel('Age')
plt.ylabel('Charges')


# Create a Pandas DataFrame
df = pd.DataFrame(data, columns=header)

# Convert specific columns to numeric data types
df['age'] = pd.to_numeric(df['age'])
df['charges'] = pd.to_numeric(df['charges'])

# Create a scatterplot matrix
scatter_matrix(df[['age', 'charges']], alpha=0.5, figsize=(8, 8), diagonal='hist')

# Create a pixel-oriented visualization (heatmap)
plt.figure(figsize=(8, 6))
plt.hist2d(ages, bmis, bins=(10, 10), cmap='viridis')
plt.colorbar()
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Pixel-Oriented Visualization of Age vs. BMI')


# Create boxplots for each attribute
plt.figure(figsize=(12, 8))

# Age Boxplot
plt.subplot(221)
plt.boxplot(ages)
plt.title('Age Boxplot')

plt.tight_layout()
plt.show()
