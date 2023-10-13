import itertools
import os


paths = ["deepdoors2/image", "deepdoors2/segmentation_mask"]

for p in paths:
    for f in os.listdir(p):
        if '(' in f:
            os.remove('/'.join([p,f]))