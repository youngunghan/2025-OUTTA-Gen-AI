import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define two distributions (P is the true distribution, Q is the model distribution)
x = np.linspace(0.01, 1, 100)  # Generate 100 points between 0.01 and 1
P = 0.6 * np.exp(-x)  # True distribution
Q = 0.4 * np.exp(-1.5 * x)  # Model distribution (slightly different shape)

# Normalize the distributions
P /= np.sum(P)  # Normalize P so that it sums to 1
Q /= np.sum(Q)  # Normalize Q so that it sums to 1

# Compute KL Divergence (using the normalized distributions)
kl_div = entropy(P, Q)  # KL Divergence between P and Q

# Visualization
plt.figure(figsize=(8, 5))  # Set the figure size
plt.plot(x, P, label="P(x) (True Distribution)", color='blue', linewidth=2)  # Plot P(x)
plt.plot(x, Q, label="Q(x) (Model Distribution)", color='orange', linestyle='--', linewidth=2)  # Plot Q(x)
plt.fill_between(x, P, Q, color='gray', alpha=0.3, label=f"KL Divergence = {kl_div:.3f}")  # Highlight the divergence area
plt.title("KL Divergence Visualization")  # Set the plot title
plt.xlabel("x")  # Label for the x-axis
plt.ylabel("Probability Density")  # Label for the y-axis
plt.legend()  # Add a legend to explain the lines
plt.show()  # Display the plot
