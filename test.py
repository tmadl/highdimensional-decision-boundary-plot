import uci_loader, numpy as np
from decisionboundaryplot import DBPlot
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.preprocessing.data import normalize
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.datasets.base import load_digits
from sklearn.manifold.isomap import Isomap
from sklearn import lda
from sklearn.decomposition.nmf import NMF
from sklearn.svm.classes import SVC

X, y = uci_loader.getdataset('wine')
if np.min(X)<0:
    X -= np.min(X)
print np.min(X)
#data = load_digits(n_class = 2)
#X = data.data
#y = data.target
y[y!=0] = 1

db = DBPlot(SVC(probability=True))#, NMF(n_components=2)) # lda.LDA(solver='eigen', n_components=2))
db.fit(X, y, training_indices=0.5)
db.plot(tune_background_model=True).show()