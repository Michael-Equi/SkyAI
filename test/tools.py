import numpy as np
import json
import sys
import requests
import imageio

# read the command line input for the path to the json file with labels
if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    print("Incorrect usage: please indicate a file of the data")
    sys.exit(1)
arr = []
with open(filepath, "r") as read_file:
    data = json.load(read_file)
    string = json.dumps(data, indent=2)
    # print(string) # for visualizing the json better
    for item in data:
        masks = item['Masks']
        for layer in masks:
            url = masks[layer]
            im = imageio.imread(url)
            arr.append(im[:, :, 0])
