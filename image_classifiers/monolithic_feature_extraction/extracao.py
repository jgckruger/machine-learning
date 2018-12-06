import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# carrega os dados
url = "mono_feat.csv"
dataframe = pd.read_csv(url)

array = dataframe.values
X = array[:,0:827]
y = array[:,827]

clf = LinearSVC(random_state=0, tol=1e-5)
sfm = SelectFromModel(clf, threshold=0.25)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]

clf=LinearSVC(random_state=0, tol=1e-5)
clf.fit(X,y)
clf.score(X,y)

clf2=LinearSVC(random_state=0, tol=1e-5)
clf2.fit(sfm.transform(X),y)
clf2.score(sfm.transform(X),y)

dados = np.array(sfm.transform(X))
y = np.array([y])
dados=np.concatenate((dados, y.T), axis=1)
print(dados.shape)
dados=pd.DataFrame(dados)
dados.to_csv('selected.csv', sep=',', encoding='utf-8', index=False)