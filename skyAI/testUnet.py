from model import *
from data import *
import sys
import imageio
import skimage.transform as trans

if not len(sys.argv) > 1:
    print("Usage error: make sure to specify the name of the image to make prediction on")

input_shape = (256, 256, 3)

model = unet()
model.load_weights("sky.hdf5")
image = model.predict(np.array([trans.resize(imageio.imread(sys.argv[1]), input_shape)]))
imageio.imwrite("predicted_mask.jpg", image[0, :, :, :])
