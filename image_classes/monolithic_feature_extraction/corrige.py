# python feature_extraction.py --image_dir ./imagens/ --output_labels=labels.txt --features ./saida.csv

import numpy as np
import pandas as pd

s = ""
for i in range(827):
    s += str(i) + ","

s += "class\n"

with open('mono_feat.csv', 'w') as f:
    f.write(s)
    with open('feat.csv', 'r') as arq:
        f.write(arq.read())

