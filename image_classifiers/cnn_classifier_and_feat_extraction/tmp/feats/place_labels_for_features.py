# python feature_extraction.py --image_dir ./imagens/ --output_labels=labels.txt --features ./saida.csv

import numpy as np
import pandas as pd



df = pd.read_csv('features.csv')
number_of_features = df.values[0].size-1


for i in range(number_of_features):
    s += str(i) + ","
s += "class\n"


with open('cnn_feat.csv', 'w') as f:
    f.write(s)
    with open('features.csv', 'r') as arq:
        f.write(arq.read())

