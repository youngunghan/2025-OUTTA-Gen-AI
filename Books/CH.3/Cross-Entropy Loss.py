import numpy as np
import matplotlib.pyplot as plt

# True label
true_label = [1, 0, 0]  # Example: the true label is [1, 0, 0] (class 1 is the correct class)

# Predicted probabilities (gradually improving)
predictions = np.linspace(0.1, 0.9, 100)  # Generate 100 predicted probabilities between 0.1 and 0.9
losses = -np.log(predictions)  # Compute cross-entropy loss: -log(predicted probability)

plt.figure(figsize=(6, 4))  # Set the figure size (width: 6, height: 4)
plt.plot(predictions, losses, label="Cross-Entropy Loss")  # Plot loss values for predicted probabilities
plt.title("Cross-Entropy Loss")  # Set the plot title
plt.xlabel("Predicted Probability for True Label")  # X-axis label: predicted probability for the correct class
plt.ylabel("Loss")  # Y-axis label: loss value

# Highlight the point where the predicted probability is 0.7
plt.axvline(x=0.7, color='r', linestyle='--', label="Good Prediction (0.7)")  
# Add a vertical dashed line at x = 0.7, representing a "good" prediction

plt.legend()  # Add a legend
plt.show()  # Display the plot
