import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import tight_layout



# Create a figure and a 3D Axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the x, y, and z values
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
x, y = np.meshgrid(x, y)
z = x**2 + y**2

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')
#tight_layout()
#plt.savefig('./plots/3D_surface_plot.png', dpi=300, bbox_inches='tight')

plt.show()