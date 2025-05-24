
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df= pd.read_csv('Data/CPExplain221014.csv')
y = df['CP score'].values


def generate_softmax_vector(df, seed=None):
    size= df.shape[0]
    if seed is not None:
        np.random.seed(seed)
    
    proportions = np.random.rand(len(y))  # values in (0, 1)
    y1 = proportions * y
    y2 = (1 - proportions) * y
    return y1, y2


# Convert date column to datetime format if it's not already
df['date'] = pd.to_datetime(df['date'])

# Sort by date (optional but recommended)
df = df.sort_values('date')
ggs, bnb= generate_softmax_vector(df, seed= 101)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['date'], df['CP score'], marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("CP score")
plt.title("CP score over time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Plot all three series
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['CP score'], label='Total-score', color='black', linewidth=2, marker='o')
plt.plot(df['date'], ggs, label='GGS-score', linestyle='--', color='blue', marker='o')
plt.plot(df['date'], bnb, label='BNB-score', linestyle=':', color='green', marker='o')

plt.xlabel("Date", fontsize=18)
plt.ylabel("CP score", fontsize=18)
plt.title("CP score split into GGS and BNB over time", fontsize=20)
plt.xticks(rotation=45, fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()
