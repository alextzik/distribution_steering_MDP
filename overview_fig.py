import numpy as np
import matplotlib.pyplot as plt

# Create two random 2D distributions
np.random.seed(0)
mean1 = [-2, 0]
cov1 = 0.5*np.array([[1, 0], [0, 1]])
mean2 = [4, 0]
cov2 = 0.5*np.array([[1, 0.5], [0.5, 1]])
dist1 = np.random.multivariate_normal(mean1, cov1, 1000)
dist2 = np.random.multivariate_normal(mean2, cov2, 1000)

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 10))

# Plot density of distributions
ax.hexbin(dist1[:, 0], dist1[:, 1], gridsize=50, cmap='Blues', alpha=0.6, mincnt=1)
ax.hexbin(dist2[:, 0], dist2[:, 1], gridsize=50, cmap='Reds', alpha=0.6, mincnt=1)

# Projection line
line_x = np.linspace(-3, 5, 100)
line_y = line_x
ax.plot(line_x, line_y, color="black", linestyle="--")
# add text "Projection Line" near the bottom right of the line
ax.text(-2, -2.5, "Projection Line", rotation=45, verticalalignment='bottom', horizontalalignment='right')

# Projections onto the line
projection1 = (dist1[:, 0] - dist1[:, 1]) / np.sqrt(2)
projection2 = (dist2[:, 0] - dist2[:, 1]) / np.sqrt(2)

# Calculate CDFs
cdf_x_vals = np.linspace(projection1.min(), projection2.max(), 1000)
cdf1 = np.array([np.mean(projection1 <= val) for val in cdf_x_vals])
cdf2 = np.array([np.mean(projection2 <= val) for val in cdf_x_vals])

# Create a secondary axis aligned with the projection line
secax = ax.inset_axes([0.1, 0.8, 0.5, 0.18]) # move more to left
secax.plot(cdf_x_vals, cdf1, color='blue', label="Projection of X")
secax.plot(cdf_x_vals, cdf2, color='red', label="Projection of Y")
secax.set_xlabel('Value')
secax.set_ylabel('CDF')
secax.legend(loc='upper left')

# Scaling factor for visualizing the CDF over the projection line
scale_factor = 1.
scaled_cdf1 = cdf1 * scale_factor
scaled_cdf2 = cdf2 * scale_factor

# Rotate and plot the CDFs on the primary axis
for x_val, cdf1_val, cdf2_val in zip(cdf_x_vals, scaled_cdf1, scaled_cdf2):
    x_proj = x_val 
    y_proj = x_val
    cdf1_x_proj = x_proj 
    cdf1_y_proj = y_proj + cdf1_val 
    cdf2_x_proj = x_proj 
    cdf2_y_proj = y_proj + cdf2_val 

    ax.plot([x_proj, cdf1_x_proj], [y_proj, cdf1_y_proj], color='blue', alpha=0.2)
    ax.plot([x_proj, cdf2_x_proj], [y_proj, cdf2_y_proj], color='red', alpha=0.2)

# Final adjustments
ax.grid(True)

plt.tight_layout()
plt.savefig("combined_density_simulated_3d_CDFs_with_axes.pdf")
plt.show()