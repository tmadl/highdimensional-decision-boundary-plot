import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition.pca import PCA
from sklearn.neighbors.unsupervised import NearestNeighbors

def DBPlot(BaseEstimator):
    """
    Heuristic approach to estimate and visualize high-dimensional decision 
    boundaries for trained binary classifiers by using black-box optimization 
    to find regions in which the classifier is maximally uncertain (0.5 prediction
    probability). The total number of keypoints representing the decision boundary
    will depend on n_connecting_keypoints and n_interpolated_keypoints.
    
    Parameters
    ----------
    estimator : BaseEstimator instance, optional (default=`RandomForestClassifier()`).
        Classifier for which the decision boundary should be plotted. Can be trained
        or untrained (in which case the fit method will train it). Must have
        probability estimates enabled (i.e. `estimator.predict_proba` must work). 
        Make sure it is possible for probability estimates to get close to 0.5 
        (more specifically, as close as specified by acceptance_threshold).
    
    dimensionality_reduction : BaseEstimator instance, optional (default=`PCA(n_components=2)`).
        Dimensionality reduction method to help plot the decision boundary in 2D. Can be trained
        or untrained (in which case the fit method will train it). Must have n_components=2. 
        Must be able to project new points into the 2D space after fitting 
        (i.e. `dimensionality_reduction.transform` must work). 
    
    acceptance_threshold : float, optional (default=0.03)
        Maximum allowed deviation from decision boundary (defined as the region 
        with 0.5 prediction probability) when accepting decision boundary keypoints
        
    n_connecting_keypoints : int, optional (default=20)
        Number of decision boundary keypoints estimated along lines connecting
        instances from two different classes (each such line must cross the 
        decision boundary at least once). 
        
    n_interpolated_keypoints : int, optional (default=50)
        Number of decision boundary keypoints interpolated between connecting
        keypoints to increase keypoint density.  
        
    n_generated_testpoints_per_keypoint : int, optional (default=20)
        Number of test points generated around decision boundary keypoints, and 
        labeled according to the specified classifier, in order to enrich and 
        validate the decision boundary plot
        
    linear_iteration_budget : int, optional (default=100)
        Maximum number of iterations the optimizer is allowed to run for each
        keypoint estimation while looking along linear trajectories
        
    hypersphere_iteration_budget : int, optional (default=300)
        Maximum number of iterations the optimizer is allowed to run for each
        keypoint estimation while looking along hypersphere surfaces
    """
    def __init__(self, estimator=RandomForestClassifier(), dimensionality_reduction=PCA(n_components=2), acceptance_threshold=0.03, n_connecting_keypoints=20, n_interpolated_keypoints=50, linear_iteration_budget=100, hypersphere_iteration_budget=300):
        if acceptance_threshold == 0:
            raise Warning("A nonzero acceptance threshold is strongly recommended so the optimizer can finish in finite time")
        if linear_iteration_budget < 2 or hypersphere_iteration_budget < 2:
            raise Exception("Invalid iteration budget")
        if dimensionality_reduction.n_components_ != 2:
            raise Exception("Dimensionality reduction method must project to two dimensions for visualization!")
        
        self.model = estimator
        self.dimensionality_reduction = dimensionality_reduction
        self.acceptance_threshold = acceptance_threshold
        self.linear_iteration_budget = linear_iteration_budget
        self.n_connecting_keypoints = n_connecting_keypoints 
        self.n_interpolated_keypoints = n_interpolated_keypoints
        self.hypersphere_iteration_budget = hypersphere_iteration_budget
        
    def setmodel(self, estimator=RandomForestClassifier()):
        """Assign model for which decision boundary should be plotted.

        Parameters
        ----------
        estimator : BaseEstimator instance, optional (default=RandomForestClassifier()).
            Classifier for which the decision boundary should be plotted. Must have
            probability estimates enabled (i.e. estimator.predict_proba must work). 
            Make sure it is possible for probability estimates to get close to 0.5 
            (more specifically, as close as specified by acceptance_threshold).
        """
        self.model = estimator
    
    def fit(self, X, y, training_indices=None):
        """Specify data to be plotted, and fit model only if required (the 
        specified model is only trained if it has not been trained yet). 

        All the input data is provided in the matrix X, and corresponding 
        binary labels (values taking 0 or 1) in the vector y 

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix containing data 

        y : array-like, shape = [n_samples]
            Labels
            
        training_indices : array-like or float, optional (default=None)
            Indices on which the model has been trained / should be trained. 
            If float, it is converted to a random sample with the specified proportion
            of the full dataset.

        Returns
        -------
        self : returns an instance of self.
        """
        
        if training_indices == None:
            train_idx = range(len(y))
        elif type(training_indices) == float:
            train_idx, test_idx = train_test_split(range(len(y)), test_size=0.5)
        else:
            train_idx = training_indices
            
        self.X = X
        self.y = y
        
        # fit model if necessary
        try:
            self.model.predict([X[0]])
        except:
            self.model.fit(X[train_idx, :], y[train_idx])
            
        # fit DR method if necessary
        try:
            self.dimensionality_reduction.transform([X[0]])
        except:
            self.dimensionality_reduction.fit(X)
        
        try:    
            self.dimensionality_reduction.transform([X[0]])
        except:
            raise Exception("Please make sure your dimensionality reduction method has an exposed transform() method! If in doubt, use PCA or Isomap")
            
        # transform data
        self.X2d = self.dimensionality_reduction.transform(self.X)
            
        # set up efficient nearest neighbor models for later use
        self.nn_model_2d_0class = NearestNeighbors(n_neighbors=1)
        self.nn_model_2d_0class.fit(self.X2d[self.y==0])
        
        self.nn_model_2d_1class = NearestNeighbors(n_neighbors=1)
        self.nn_model_2d_1class.fit(self.X2d[self.y==1])