import uci_loader
from decisionboundaryplot import DBPlot
from sklearn.ensemble.forest import RandomForestClassifier

X, y = uci_loader.getdataset('wine')

db = DBPlot(RandomForestClassifier())
db.fit(X, y, training_indices=0.5)
db.plot().show()