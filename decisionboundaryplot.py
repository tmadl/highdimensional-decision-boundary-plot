import numpy as np, matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from sklearn.decomposition.pca import PCA
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.neighbors.classification import KNeighborsClassifier
import nlopt, random as rnd
from scipy.spatial.distance import euclidean

def DBPlot(BaseEstimator):
    """
    Heuristic approach to estimate and visualize high-dimensional decision 
    boundaries for trained binary classifiers by using black-box optimization 
    to find regions in which the classifier is maximally uncertain (0.5 prediction
    probability). The total number of keypoints representing the decision boundary
    will depend on n_connecting_keypoints and n_interpolated_keypoints.
    
    Parameters
    ----------
    estimator : BaseEstimator instance, optional (default=`KNeighborsClassifier(n_neighbors=2)`).
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
        
    verbose: bool, optional (default=True)
        Verbose output
    """
    def __init__(self, estimator=KNeighborsClassifier(n_neighbors=2), dimensionality_reduction=PCA(n_components=2), acceptance_threshold=0.03, n_connecting_keypoints=20, n_interpolated_keypoints=50, linear_iteration_budget=100, hypersphere_iteration_budget=300, verbose=True):
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
        self.verbose = verbose
        
        self.decision_boundary_points = []
        self.decision_boundary_points_2d = []
        
    def setmodel(self, estimator=KNeighborsClassifier(n_neighbors=2)):
        """Assign model for which decision boundary should be plotted.

        Parameters
        ----------
        estimator : BaseEstimator instance, optional (default=KNeighborsClassifier(n_neighbors=2)).
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
            
        # decision boundary distance : distance from the region with maximal uncertainty (0.5 prediction probability)
        self.decision_boundary_distance = lambda x, grad=0: np.abs(0.5-self.model.predict_proba([x])[0][1])
            
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
        self.X2d_xmin, self.X2d_xmax = np.min(self.X2d[:,0]), np.max(self.X2d[:,0])
        self.X2d_ymin, self.X2d_ymax = np.min(self.X2d[:,1]), np.max(self.X2d[:,1])
        
        self.majorityclass = 0 if list(y).count(0) > list(y).count(1) else 1
        minority_idx, majority_idx = y==self.minorityclass, y==self.majorityclass
        self.Xminor, self.Xmajor = X[minority_idx], X[majority_idx]
        self.Xminor2d, self.Xmajor2d = self.X2d[minority_idx], self.X2d[majority_idx]
            
        # set up efficient nearest neighbor models for later use
        self.nn_model_2d_majorityclass = NearestNeighbors(n_neighbors=2)
        self.nn_model_2d_majorityclass.fit(self.X2d[self.y==self.majorityclass])
        
        self.nn_model_2d_minorityclass = NearestNeighbors(n_neighbors=2)
        self.nn_model_2d_minorityclass.fit(self.X2d[self.y==self.minorityclass])
        
        # step 1. look for decision boundary points between corners of majority & minority class distribution
        minority_corner_idx, majority_corner_idx = [], []
        for extremum1 in [np.min, np.max]:
            for extremum2 in [np.min, np.max]:
                d, idx = self.nn_model_2d_minorityclass.kneighbors([[extremum1(self.Xminor2d[:,0]), extremum2(self.Xminor2d[:,0])]])
                minority_corner_idx.append(idx[0][0])
                d, idx = self.nn_model_2d_minorityclass.kneighbors([[extremum1(self.Xmajor2d[:,1]), extremum2(self.Xmajor2d[:,1])]])
                majority_corner_idx.append(idx[0][0])
        
        self._linear_decision_boundary_optimization(minority_corner_idx, majority_corner_idx, all_combinations=True)
        
        # step 2. look for decision boundary points on lines connecting randomly sampled points of majority & minority class
        from_idx = list(rnd.sample(np.arange(len(self.Xminor)), self.n_connecting_keypoints))
        to_idx = list(rnd.sample(np.arange(len(self.Xmajor)), self.n_connecting_keypoints))
        
        self._linear_decision_boundary_optimization(from_idx, to_idx, all_combinations=False)
        

    def _linear_decision_boundary_optimization(self, from_idx, to_idx, all_combinations=True, retry_neighbor_if_failed=True):
        """Use global optimization to locate the decision boundary along lines
        defined by instances from_idx and to_idx in the dataset (from_idx and to_idx
        have to contain indices from distinct classes to guarantee the existence of
        a decision boundary between them!)
        """
        optimizer = self._get_optimizer()
        
        retries = 4 if retry_neighbor_if_failed else 1
        for i in range(len(from_idx)):
            n = range(len(to_idx)) if all_combinations else 1
            for j in range(n):
                from_i = from_idx[i]
                to_i = to_idx[j] if all_combinations else to_idx[i]
                for k in range(retries):
                    if k == 0:
                        fromPoint = self.Xminor[from_i]
                        toPoint = self.Xmajor[to_i]
                    else:
                        # first attempt failed, try nearest neighbors of source and destination point instead
                        d, idx = self.nn_model_2d_minorityclass.kneighbors([self.Xminor2d[from_i]])
                        fromPoint = self.Xminor[idx[0][k/2]]
                        d, idx = self.nn_model_2d_minorityclass.kneighbors([self.Xmajor2d[to_i]])
                        toPoint = self.Xmajor[idx[0][k%2]]
                    
                    if euclidean(fromPoint, toPoint) == 0:
                        break # no decision boundary between equivalent points
                
                    def objective(l, grad=0):
                        # interpolate between source and destionation; calculate distance from decision boundary
                        X = fromPoint + l * (toPoint-fromPoint)  
                        return self.decision_boundary_distance(X)
                    
                    optimizer.set_min_objective()
                    cL = optimizer.optimize([rnd.random()])
                    cX = fromPoint + cL * (toPoint-fromPoint)
                    
                    if self.decision_boundary_distance(cX) <= self.acceptance_threshold:
                        cX2d = self.dimensionality_reduction.transform([cX])[0]
                        if cX2d[0] >= self.X2d_xmin and cX2d[0] <= self.X2d_xmax and cX2d[1] >= self.X2d_ymin and cX2d[1] <= self.X2d_ymax: 
                            self.decision_boundary_points.append(cX)
                            self.decision_boundary_points_2d.append(cX2d)
                            print i*len(from_idx)+j,"/",len(from_idx)*n, ": New decision boundary keypoint found using linear optimization!"
                        else:
                            print i*len(from_idx)+j,"/",len(from_idx)*n, ": Rejected decision boundary keypoint (outside of plot area)"
        
    def _get_optimizer(self, D=1, iteration_budget=None):
        """Utility function creating an NLOPT optimizer with default
        parameters depending on this objects parameters
        """
        if iteration_budget == None:
            iteration_budget = self.linear_iteration_budget
        
        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, D)
        opt.set_stopval(self.acceptance_threshold/10.0)
        opt.set_ftol_rel(1e-4)
        opt.set_maxeval(iteration_budget)
        opt.set_lower_bounds(0)
        opt.set_upper_bounds(1)
        
        return opt