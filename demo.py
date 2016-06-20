import uci_loader, numpy as np
from decisionboundaryplot import DBPlot
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.datasets.base import load_digits
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from numpy.random.mtrand import permutation
from sklearn.svm.classes import SVC
from sklearn.neighbors.classification import KNeighborsClassifier

if __name__ == "__main__":
    # load data
    X, y = uci_loader.getdataset('wine')
    #data = load_digits(n_class = 2)
    #X,y = data.data, data.target
    y[y!=0] = 1
    
    # shuffle data
    random_idx = permutation(np.arange(len(y)))
    X = X[random_idx]
    y = y[random_idx]
    
    # create model
    model = LogisticRegression(C=1)
    #model = RandomForestClassifier(n_estimators=10)
    
    # plot high-dimensional decision boundary
    db = DBPlot(model)
    db.penalties_enabled = False
    db.fit(X, y, training_indices=0.5)
    db.plot(plt, generate_testpoints=False) # set generate_testpoints=False to speed up plotting
    plt.tight_layout(0)
    plt.show()
    
    #plot learning curves for comparison
    N = 10
    train_sizes, train_scores, test_scores = \
        learning_curve(model, X, y, cv=5, train_sizes=np.linspace(.2, 1.0, N))
    plt.errorbar(train_sizes, np.mean(train_scores, axis=1), np.std(train_scores, axis=1)/np.sqrt(N))
    plt.errorbar(train_sizes, np.mean(test_scores, axis=1), np.std(test_scores, axis=1)/np.sqrt(N), c='r')
    plt.legend(["Accuracies on training set", "Accuracies on test set"])
    plt.xlabel("Proportion of dataset used for training")
    plt.title(str(model))
    plt.tight_layout(0)
    plt.show()