import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

# Gaussian distribution
x = np.linspace(-4, 4, 1000)  # Generate 1000 points between -4 and 4 (X-axis range)
mean, std = 0, 1  # Set the mean and standard deviation to 0 and 1, respectively
pdf = norm.pdf(x, mean, std)  # Calculate the Gaussian probability density function (PDF)

plt.figure(figsize=(10, 4))  # Set the overall figure size (width: 10, height: 4)

# Plot for Gaussian distribution
plt.subplot(1, 2, 1)  # Create the first subplot in a 1-row, 2-column layout
plt.plot(x, pdf, label="Gaussian PDF")  # Plot the Gaussian PDF
plt.title("Gaussian Distribution")  # Set the title for the plot
plt.xlabel("x")  # Label for the X-axis
plt.ylabel("Density")  # Label for the Y-axis
plt.legend()  # Add a legend

# Bernoulli distribution
p = 0.7  # Set the success probability to 0.7
x = [0, 1]  # Possible values for a Bernoulli distribution (0 and 1)
pmf = bernoulli.pmf(x, p)  # Calculate the Bernoulli probability mass function (PMF)

# Plot for Bernoulli distribution
plt.subplot(1, 2, 2)  # Create the second subplot in a 1-row, 2-column layout
plt.bar(x, pmf, color='orange', label="Bernoulli PMF")  # Plot the PMF as a bar graph
plt.title("Bernoulli Distribution")  # Set the title for the plot
plt.xlabel("x")  # Label for the X-axis
plt.ylabel("Probability")  # Label for the Y-axis
plt.xticks([0, 1])  # Set the X-axis ticks to 0 and 1
plt.legend()  # Add a legend

plt.tight_layout()  # Adjust spacing between subplots
plt.show()  # Display the plots
