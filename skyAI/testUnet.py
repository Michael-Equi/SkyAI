from model import *
from data import *
import sys
import imageio

if not len(sys.argv) > 1:
	print("Usage error: make sure to specify the name of the image to make prediction on")

model = unet()
model.load_weights("sky.hdf5")
image = model.predict(np.array([imageio.imread(sys.argv[1])]))
imageio.imwrite("predicted_mask.jpg", image[0,:,:,:])


