import numpy as np
import matplotlib.pyplot as plt

# Real data distribution
real_data = np.random.normal(loc=0, scale=1, size=1000)
# Simulates real data using a normal distribution with mean 0 and standard deviation 1
# This represents the target distribution that the GAN generator tries to replicate.

# Initial generator data distribution (before training)
fake_data_initial = np.random.uniform(-2, 2, size=1000)
# Simulates data from the generator before training, using a uniform distribution
# The generator starts with a poorly matched distribution.

# Trained generator data distribution (after training)
fake_data_trained = np.random.normal(loc=0, scale=1, size=1000)
# Simulates data from the generator after training, using a normal distribution
# After sufficient training, the generator should produce data similar to the real data distribution.

# Visualization
plt.figure(figsize=(10, 5))  # Set the figure size (width: 10, height: 5)

# Plot the histogram of real data
plt.hist(real_data, bins=30, alpha=0.6, label="Real Data", color='blue')
# Use 30 bins to group data values and give the histogram transparency (alpha=0.6)

# Plot the histogram of generator data before training
plt.hist(fake_data_initial, bins=30, alpha=0.6, label="Fake Data (Before Training)", color='red')

# Plot the histogram of generator data after training
plt.hist(fake_data_trained, bins=30, alpha=0.6, label="Fake Data (After Training)", color='green')

# Add plot title and labels
plt.title("GAN Training: Real vs Fake Data")  # Title of the plot
plt.xlabel("Value")  # X-axis label
plt.ylabel("Frequency")  # Y-axis label

# Add a legend to distinguish between the three datasets
plt.legend()

# Show the plot
plt.show()
