import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load sample Titanic dataset from seaborn (similar to Kaggle dataset)
df = sns.load_dataset('titanic')

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Survival Count
sns.countplot(data=df, x='survived', ax=axes[0, 0])
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_xlabel("Survived")
axes[0, 0].set_ylabel("Count")

# Plot 2: Survival by Gender
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0, 1])
axes[0, 1].set_title("Survival by Gender")
axes[0, 1].set_xlabel("Sex")
axes[0, 1].set_ylabel("Count")

# Plot 3: Age Distribution
df['age'].plot.hist(bins=30, edgecolor='black', ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Frequency")

# Plot 4: Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title("Correlation Heatmap")

# Save the final image
output_path = "Titanic_EDA_Charts_CST_DS_02.png"
plt.tight_layout()
plt.savefig(output_path)
plt.close()
