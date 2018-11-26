import cv2
import numpy as np
import pandas as pd
import re
import math

from skimage import feature
from imutils import paths

def label(n):
    return {
        0: 'c1',
        1: 'c2',
        2: 'c3',
        3: 'c4',
        4: 'c5',
        5: 'c6',
        6: 'c7',
        7: 'c8',
        8: 'c9',
        9: 'c10'
    }[n]

X = []
for arquivo in paths.list_images('images'):
    imagem = cv2.imread(arquivo)
    altura, largura, _ = imagem.shape
    classe = math.floor(int(re.sub("\D", "", arquivo.split("/")[1]))/100)
    print("classe =",  classe)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    rgb   = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        
    r_histograma = cv2.calcHist([rgb], [0], None, [256], [0, 256])/(altura*largura)
    g_histograma = cv2.calcHist([rgb], [1], None, [256], [0, 256])/(altura*largura)
    b_histograma = cv2.calcHist([rgb], [2], None, [256], [0, 256])/(altura*largura)
    
    lbp = feature.local_binary_pattern(cinza, 59, 1, method="uniform")
    (lbp_histograma, _) = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    lbp_histograma = lbp_histograma.astype("float")
    lbp_histograma /= (lbp_histograma.sum())

    X_image = [r_histograma, g_histograma, b_histograma, lbp_histograma]
    
    X_image_aux = []
    for aux in X_image:
        X_image_aux = np.append(X_image_aux, np.ravel(aux))
    
    X_image = [i for i in X_image_aux]
    X_image.append(label(classe))
    
    X.append(X_image)
df = pd.DataFrame(X)
df.to_csv('feat.csv', header=False, index=False)
    
    


