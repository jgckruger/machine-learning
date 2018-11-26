# python feature_extraction.py --image_dir ./imagens/ --output_labels=labels.txt --features ./saida.csv

import numpy as np
import pandas as pd

for i in range(2048):
    s += str(i) + ","

s += "class\n"

with open('cnn_features.csv', 'w') as f:
    f.write(s)
    with open('saida.csv', 'r') as arq:
        f.write(arq.read())

