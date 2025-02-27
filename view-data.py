import numpy as np

datapath = "data/archive/youtube_faces_with_keypoints_full_1/youtube_faces_with_keypoints_full_1/Aaron_Eckhart_0.npz"

data = np.load(datapath)
imgs = data['colorImages']
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure and axis
fig, ax = plt.subplots()
img_display = ax.imshow(imgs[:, :, :, 0])  # Initial frame
ax.axis('off')  # Hide axis

# Update function for animation
def update(frame):
    img_display.set_array(imgs[:, :, :, frame])  # Update image
    return [img_display]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=79, interval=100)  # 100ms per frame

plt.show()


