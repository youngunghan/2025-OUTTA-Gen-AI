import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('image.jpg')
plt.imshow(img)
plt.show()