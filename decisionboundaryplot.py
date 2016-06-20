import numpy as np, matplotlib.pyplot as mplt
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split
from sklearn.decomposition.pca import PCA
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.neighbors.classification import KNeighborsClassifier
import nlopt, random as rnd
from scipy.spatial.distance import euclidean, squareform, pdist
from utils import minimum_spanning_tree, polar_to_cartesian
from sklearn.grid_search import GridSearchCV
from sklearn.svm.classes import SVC

class DBPlot(BaseEstimator):
    """
    Heuristic approach to estimate and visualize high-dimensional decision 
    boundaries for trained binary classifiers by using black-box optimization 
    to find regions in which the classifier is maximally uncertain (0.5 prediction
    probability). The total number of keypoints representing the decision boundary
    will depend on n_connecting_keypoints and n_interpolated_keypoints. Reduce
    either or both to reduce runtime.
    
    Parameters
    ----------
    estimator : BaseEstimator instance, optional (default=`KNeighborsClassifier(n_neighbors=10)`).
        Classifier for which the decision boundary should be plotted. Can be trained
        or untrained (in which case the fit method will train it). Must have
        probability estimates enabled (i.e. `estimator.predict_proba` must work). 
        Make sure it is possible for probability estimates to get close to 0.5 
        (more specifically, as close as specified by acceptance_threshold) - this usally
        requires setting an even number of neighbors, estimators etc.
    
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
    def __init__(self, estimator=KNeighborsClassifier(n_neighbors=10), dimensionality_reduction=PCA(n_components=2), acceptance_threshold=0.03, n_connecting_keypoints=20, n_interpolated_keypoints=50, n_generated_testpoints_per_keypoint=20, linear_iteration_budget=100, hypersphere_iteration_budget=300, verbose=True):
        if acceptance_threshold == 0:
            raise Warning("A nonzero acceptance threshold is strongly recommended so the optimizer can finish in finite time")
        if linear_iteration_budget < 2 or hypersphere_iteration_budget < 2:
            raise Exception("Invalid iteration budget")
        
        self.classifier = estimator
        self.dimensionality_reduction = dimensionality_reduction
        self.acceptance_threshold = acceptance_threshold
        self.linear_iteration_budget = linear_iteration_budget
        self.n_connecting_keypoints = n_connecting_keypoints 
        self.n_interpolated_keypoints = n_interpolated_keypoints
        self.n_generated_testpoints_per_keypoint = n_generated_testpoints_per_keypoint
        self.hypersphere_iteration_budget = hypersphere_iteration_budget
        self.verbose = verbose
        
        self.decision_boundary_points = []
        self.decision_boundary_points_2d = []
        self.X_testpoints = []
        self.y_testpoints = []
        self.background = []
        self.steps = 4
        
        self.hypersphere_max_retry_budget = 20
        self.penalties_enabled = True
        self.random_gap_selection = False
        
    def setclassifier(self, estimator=KNeighborsClassifier(n_neighbors=10)):
        """Assign classifier for which decision boundary should be plotted.

        Parameters
        ----------
        estimator : BaseEstimator instance, optional (default=KNeighborsClassifier(n_neighbors=10)).
            Classifier for which the decision boundary should be plotted. Must have
            probability estimates enabled (i.e. estimator.predict_proba must work). 
            Make sure it is possible for probability estimates to get close to 0.5 
            (more specifically, as close as specified by acceptance_threshold).
        """
        self.classifier = estimator
    
    def fit(self, X, y, training_indices=None):
        """Specify data to be plotted, and fit classifier only if required (the 
        specified clasifier is only trained if it has not been trained yet). 

        All the input data is provided in the matrix X, and corresponding 
        binary labels (values taking 0 or 1) in the vector y 

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix containing data 

        y : array-like, shape = [n_samples]
            Labels
            
        training_indices : array-like or float, optional (default=None)
            Indices on which the classifier has been trained / should be trained. 
            If float, it is converted to a random sample with the specified proportion
            of the full dataset.

        Returns
        -------
        self : returns an instance of self.
        """
        if set(np.array(y, dtype=int).tolist()) != set([0,1]):
            raise Exception("Currently only implemented for binary classification. Make sure you pass in two classes (0 and 1)")
        
        if training_indices == None:
            train_idx = range(len(y))
        elif type(training_indices) == float:
            train_idx, test_idx = train_test_split(range(len(y)), test_size=0.5)
        else:
            train_idx = training_indices
            
        self.X = X
        self.y = y
        self.train_idx = train_idx
        #self.test_idx = np.setdiff1d(np.arange(len(y)), self.train_idx, assume_unique=False)
        self.test_idx = list(set(range(len(y))).difference(set(self.train_idx)))
        
        # fit classifier if necessary
        try:
            self.classifier.predict([X[0]])
        except:
            self.classifier.fit(X[train_idx, :], y[train_idx])
            
        self.y_pred = self.classifier.predict(self.X)
            
        # fit DR method if necessary
        try:
            self.dimensionality_reduction.transform([X[0]])
        except:
            self.dimensionality_reduction.fit(X, y)
        
        try:    
            self.dimensionality_reduction.transform([X[0]])
        except:
            raise Exception("Please make sure your dimensionality reduction method has an exposed transform() method! If in doubt, use PCA or Isomap")
            
        # transform data
        self.X2d = self.dimensionality_reduction.transform(self.X)
        self.mean_2d_dist = np.mean(pdist(self.X2d))
        self.X2d_xmin, self.X2d_xmax = np.min(self.X2d[:,0]), np.max(self.X2d[:,0])
        self.X2d_ymin, self.X2d_ymax = np.min(self.X2d[:,1]), np.max(self.X2d[:,1])
        
        self.majorityclass = 0 if list(y).count(0) > list(y).count(1) else 1
        self.minorityclass = 1 - self.majorityclass
        minority_idx, majority_idx = np.where(y==self.minorityclass)[0], np.where(y==self.majorityclass)[0]
        self.Xminor, self.Xmajor = X[minority_idx], X[majority_idx]
        self.Xminor2d, self.Xmajor2d = self.X2d[minority_idx], self.X2d[majority_idx]
            
        # set up efficient nearest neighbor models for later use
        self.nn_model_2d_majorityclass = NearestNeighbors(n_neighbors=2)
        self.nn_model_2d_majorityclass.fit(self.X2d[majority_idx, :])
        
        self.nn_model_2d_minorityclass = NearestNeighbors(n_neighbors=2)
        self.nn_model_2d_minorityclass.fit(self.X2d[minority_idx, :])
        
        # step 1. look for decision boundary points between corners of majority & minority class distribution
        minority_corner_idx, majority_corner_idx = [], []
        for extremum1 in [np.min, np.max]:
            for extremum2 in [np.min, np.max]:
                _, idx = self.nn_model_2d_minorityclass.kneighbors([[extremum1(self.Xminor2d[:,0]), extremum2(self.Xminor2d[:,1])]])
                minority_corner_idx.append(idx[0][0])
                _, idx = self.nn_model_2d_majorityclass.kneighbors([[extremum1(self.Xmajor2d[:,0]), extremum2(self.Xmajor2d[:,1])]])
                majority_corner_idx.append(idx[0][0])
        
        # optimize to find new db keypoints between corners
        self._linear_decision_boundary_optimization(minority_corner_idx, majority_corner_idx, all_combinations=True, step=1)
        
        # step 2. look for decision boundary points on lines connecting randomly sampled points of majority & minority class
        from_idx = list(rnd.sample(np.arange(len(self.Xminor)), self.n_connecting_keypoints))
        to_idx = list(rnd.sample(np.arange(len(self.Xmajor)), self.n_connecting_keypoints))
        
        # optimize to find new db keypoints between minority and majority class
        self._linear_decision_boundary_optimization(from_idx, to_idx, all_combinations=False, step=2)
                
        if len(self.decision_boundary_points_2d)<2:
            print("Failed to find initial decision boundary. Retrying... If this keeps happening, increasing the acceptance threshold might help. Also, make sure the classifier is able to find a point with 0.5 prediction probability (usually requires an even number of estimators/neighbors/etc).")
            return self.fit(X, y, training_indices)
        
        # step 3. look for decision boundary points between already known db points that are too distant (search on connecting line first, then on surrounding hypersphere surfaces)
        edges, gap_distances, gap_probability_scores = self._get_sorted_db_keypoint_distances() # find gaps
        self.nn_model_decision_boundary_points = NearestNeighbors(n_neighbors=2)
        self.nn_model_decision_boundary_points.fit(self.decision_boundary_points)
        
        i = 0
        retries = 0
        while i < self.n_interpolated_keypoints:
            if self.verbose:
                print "Step 3/"+str(self.steps)+":",i,"/",self.n_interpolated_keypoints
            if self.random_gap_selection:
                # randomly sample from sorted DB keypoint gaps?
                gap_idx = np.random.choice(len(gap_probability_scores), 1, p=gap_probability_scores)[0]
            else:
                # get largest gap
                gap_idx = 0
            fromPoint = self.decision_boundary_points[edges[gap_idx][0]]
            toPoint = self.decision_boundary_points[edges[gap_idx][1]]
            
            # optimize to find new db keypoint along line connecting two db keypoints with large gap
            dbPoint = self._find_decision_boundary_along_line(fromPoint, toPoint, penalizeTangentDistance=self.penalties_enabled)
            
            if self.decision_boundary_distance(dbPoint) > self.acceptance_threshold:
                if self.verbose:
                    print "No good solution along straight line - trying to find decision boundary on hypersphere surface around known decision boundary point"
                
                R = euclidean(fromPoint, toPoint)/2.0 # hypersphere radius half the distance between from and to db keypoints
                if rnd.random > 0.5: # search around either source or target keypoint, with 0.5 probability, hoping to find decision boundary in between
                    fromPoint = toPoint
                    
                # optimize to find new db keypoint on hypersphere surphase around known keypoint
                dbPoint = self._find_decision_boundary_on_hypersphere(fromPoint, R)
                if self.decision_boundary_distance(dbPoint) <= self.acceptance_threshold:
                    dbPoint2d = self.dimensionality_reduction.transform([dbPoint])[0]
                    self.decision_boundary_points.append(dbPoint)
                    self.decision_boundary_points_2d.append(dbPoint2d)
                    i += 1
                    retries = 0
                else:
                    retries += 1
                    if retries > self.hypersphere_max_retry_budget:
                        i += 1
                        print "Found point is too distant from decision boundary (",self.decision_boundary_distance(dbPoint),"), but retry budget exceeded (",self.hypersphere_max_retry_budget,")"
                    elif self.verbose:
                        print "Found point is too distant from decision boundary (",self.decision_boundary_distance(dbPoint),") retrying..."
                    
            else:
                dbPoint2d = self.dimensionality_reduction.transform([dbPoint])[0]
                self.decision_boundary_points.append(dbPoint)
                self.decision_boundary_points_2d.append(dbPoint2d)
                i += 1
                retries = 0
                
            edges, gap_distances, gap_probability_scores = self._get_sorted_db_keypoint_distances() # reload gaps
        
        self.decision_boundary_points = np.array(self.decision_boundary_points)
        self.decision_boundary_points_2d = np.array(self.decision_boundary_points_2d)
        
        if self.verbose:
            print "Done fitting! Found ",len(self.decision_boundary_points),"decision boundary keypoints."
        
        return self

    def plot(self, plt=None, generate_background=True, tune_background_model=False, background_resolution=100):
        """Plots the dataset and the identified decision boundary in 2D.
        (If you wish to create custom plots, get the data using generate_plot() and plot it manually)  
        
        Parameters
        ----------
        plt : matplotlib.pyplot or axis object (default=matplotlib.pyplot)
            Object to be plotted on
        
        generate_background : boolean, optional (default=True)
            Whether to generate faint background plot (using prediction probabilities
            of a fitted suppor vector machine, trained on generated test points) 
            to aid visualization
            
        tune_background_model : boolean, optional (default=False)
            Whether to tune the parameters of the support vector machine generating
            the background 
            
        background_resolution : int, optional (default=100)
            Desired resolution (height and width) of background to be generated
            
        Returns
        -------
        plt : The matplotlib.pyplot or axis object which has been passed in, after
        plotting the data and decision boundary on it. (plt.show() is NOT called
        and will be required)
        """
        if plt == None:
            plt = mplt
        
        if len(self.background) == 0:
            self.generate_plot(generate_background, tune_background_model, background_resolution)
        
        if generate_background:
            plt.imshow(np.flipud(self.background), extent=[self.X2d_xmin, self.X2d_xmax, self.X2d_ymin, self.X2d_ymax], cmap="GnBu", alpha=0.33)
        
        # decision boundary
        plt.scatter(self.decision_boundary_points_2d[:,0], self.decision_boundary_points_2d[:,1], 600, c='c', marker='p')
        # generated test points
        plt.scatter(self.X_testpoints_2d[:,0], self.X_testpoints_2d[:,1], 20, c=['g' if i else 'b' for i in self.y_testpoints], alpha=0.5)
        
        # training data
        plt.scatter(self.X2d[self.train_idx,0], self.X2d[self.train_idx,1], 150, \
                    facecolor=['g' if i else 'b' for i in self.y[self.train_idx]], \
                    edgecolor = ['g' if self.y_pred[self.train_idx[i]]==self.y[self.train_idx[i]]==1 \
                                        else ('b' if self.y_pred[self.train_idx[i]]==self.y[self.train_idx[i]]==0 else 'r') \
                                for i in range(len(self.train_idx))], linewidths=5)
        # testing data
        plt.scatter(self.X2d[self.test_idx,0], self.X2d[self.test_idx,1], 150, \
                    facecolor=['g' if i else 'b' for i in self.y[self.test_idx]], \
                    edgecolor = ['g' if self.y_pred[self.test_idx[i]]==self.y[self.test_idx[i]]==1 \
                                        else ('b' if self.y_pred[self.test_idx[i]]==self.y[self.test_idx[i]]==0 else 'r') \
                                for i in range(len(self.test_idx))], linewidths=5, marker='s')
        
        # label data points with their indices
        for i in range(len(self.X2d)):
            plt.text(self.X2d[i,0]+(self.X2d_xmax-self.X2d_xmin)*0.5e-2, self.X2d[i,1]+(self.X2d_ymax-self.X2d_ymin)*0.5e-2, str(i), size=8)

        plt.legend(["Estimated decision boundary keypoints", "Generated test data around decision boundary", "Actual data (training set)", "Actual data (test set)"], loc="lower right")
        
        # decision boundary keypoints, in case not visible in background
        plt.scatter(self.decision_boundary_points_2d[:,0], self.decision_boundary_points_2d[:,1], 600, c='c', marker='p', alpha=0.1)
        plt.scatter(self.decision_boundary_points_2d[:,0], self.decision_boundary_points_2d[:,1], 30, c='c', marker='p', edgecolor='c', alpha=0.8)

        # minimum spanning tree through decision boundary keypoints
        D = pdist(self.decision_boundary_points_2d)
        edges = minimum_spanning_tree(squareform(D))
        for e in edges:
            plt.plot([self.decision_boundary_points_2d[e[0],0], self.decision_boundary_points_2d[e[1],0]], [self.decision_boundary_points_2d[e[0],1], self.decision_boundary_points_2d[e[1],1]], '--c', linewidth=4)
            plt.plot([self.decision_boundary_points_2d[e[0],0], self.decision_boundary_points_2d[e[1],0]], [self.decision_boundary_points_2d[e[0],1], self.decision_boundary_points_2d[e[1],1]], '--k', linewidth=1)
            
        if self.verbose:
            print "Plot successfully generated! Don't forget to call the show() method to display it"
            
        return plt
        
    def generate_plot(self, generate_background=True, tune_background_model=False, background_resolution=100):
        """Generates and returns arrays for visualizing the dataset and the 
        identified decision boundary in 2D. 
        
        Parameters
        ----------
        generate_background : boolean, optional (default=True)
            Whether to generate faint background plot (using prediction probabilities
            of a fitted suppor vector machine, trained on generated test points) 
            to aid visualization
            
        tune_background_model : boolean, optional (default=False)
            Whether to tune the parameters of the support vector machine generating
            the background 
            
        background_resolution : int, optional (default=100)
            Desired resolution (height and width) of background to be generated 
        
        Returns
        -------
        decision_boundary_points_2d : array
            Array containing points in the dimensionality-reduced 2D space which 
            are very close to the true decision boundary
            
        X_testpoints_2d : array
            Array containing generated test points in the dimensionality-reduced 
            2D space which surround the decision boundary and can be used for 
            visual feedback to estimate which area would be assigned which class
            
        y_testpoints : array
            Classifier predictions for each of the generated test points
            
        background: array
            Generated background image showing prediction probabilities of the 
            classifier in each region (only returned if generate_background is set
            to True!)
            
        """
        if len(self.decision_boundary_points) == 0:
            raise Exception("Please call the fit method first!")
        
        if len(self.X_testpoints) == 0:
            if self.verbose:
                print "Generating test points around decision boundary..."
            self._generate_testpoints()
            
            if generate_background:
                if tune_background_model:
                    params = {'C': np.power(10, np.linspace(0,2,2)), 'gamma': np.power(10, np.linspace(-2,0,2))}
                    grid = GridSearchCV(SVC(), params, n_jobs=-1).fit(np.vstack((self.X2d[self.train_idx], self.X_testpoints_2d)), np.hstack((self.y[self.train_idx], self.y_testpoints)))
                    bestparams = grid.best_params_
                else:
                    bestparams = {'C':1, 'gamma':1}
                self.background_model = SVC(probability=True, C=bestparams['C'], gamma=bestparams['gamma']).fit(np.vstack((self.X2d[self.train_idx], self.X_testpoints_2d)), np.hstack((self.y[self.train_idx], self.y_testpoints)))
                xx, yy = np.meshgrid(np.linspace(self.X2d_xmin, self.X2d_xmax, background_resolution), np.linspace(self.X2d_ymin, self.X2d_ymax, background_resolution))
                Z = self.background_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,0]
                Z = Z.reshape((background_resolution,background_resolution))
                self.background = Z
        
        if generate_background:
            return self.decision_boundary_points_2d, self.X_testpoints_2d, self.y_testpoints, Z
        else:
            return self.decision_boundary_points_2d, self.X_testpoints_2d, self.y_testpoints
        
    def _generate_testpoints(self, tries=100):
        """Generate random test points around decision boundary keypoints 
        """
        nn_model = NearestNeighbors(n_neighbors=3)
        nn_model.fit(self.decision_boundary_points)
        
        nn_model_2d = NearestNeighbors(n_neighbors=2)
        nn_model_2d.fit(self.decision_boundary_points_2d)
        maxRadius = 2*np.max([nn_model_2d.kneighbors([self.decision_boundary_points_2d[i]])[0][0][1] for i in range(len(self.decision_boundary_points_2d))])
        
        self.X_testpoints = np.zeros((0, self.X.shape[1]))
        self.y_testpoints = []
        for i in range(len(self.decision_boundary_points)):
            if self.verbose:
                print "Generating testpoint ",i,"/",len(self.decision_boundary_points)
            testpoints = np.zeros((0, self.X.shape[1]))
            # generate Np points in Gaussian around decision_boundary_points[i] with radius depending on the distance to the next point
            d, idx = nn_model.kneighbors([self.decision_boundary_points[i]])
            radius = d[0][1] if d[0][1] != 0 else d[0][2]
            if radius == 0:
                radius = np.mean(pdist(self.decision_boundary_points))
                
            # find at least one point in each class
            classes = []
            for try_i in range(tries):
                cRadius = radius
                for try_j in range(tries):
                    testpoint = np.random.normal(self.decision_boundary_points[i], radius, (1,self.X.shape[1]))
                    try:
                        testpoint2d = self.dimensionality_reduction.transform(testpoint)[0]
                    except: # DR can fail e.g. if NMF gets negative values
                        testpoint = None
                        continue
                    if euclidean(testpoint2d, self.decision_boundary_points_2d[i]) <= maxRadius:
                        break
                    cRadius /= 2.0
                if testpoint != None:
                    testpoint_class = self.classifier.predict(testpoint)[0]
                    if len(classes) == 0:
                        testpoints = np.vstack((testpoints, testpoint))
                        classes.append(testpoint_class)
                    elif classes[0] != testpoint_class:
                        testpoints = np.vstack((testpoints, testpoint))
                        break
                
            # add other points
            for j in range(self.n_generated_testpoints_per_keypoint - 2):
                cRadius = radius
                for try_i in range(tries):
                    testpoint = np.random.normal(self.decision_boundary_points[i], radius, (1,self.X.shape[1]))
                    try:
                        testpoint2d = self.dimensionality_reduction.transform(testpoint)[0]
                    except: # DR can fail e.g. if NMF gets negative values
                        testpoint = None
                        continue
                    if euclidean(testpoint2d, self.decision_boundary_points_2d[i]) <= maxRadius:
                        break
                    cRadius /= 2.0
                if testpoint != None:
                    testpoints = np.vstack((testpoints, testpoint))
                
            self.X_testpoints = np.vstack((self.X_testpoints, testpoints))
            self.y_testpoints = np.hstack((self.y_testpoints, self.classifier.predict(testpoints)))
            self.X_testpoints_2d = self.dimensionality_reduction.transform(self.X_testpoints)
            
        idx_within_bounds = np.where((self.X_testpoints_2d[:,0]>=self.X2d_xmin)&(self.X_testpoints_2d[:,0]<=self.X2d_xmax)\
                              &(self.X_testpoints_2d[:,1]>=self.X2d_ymin)&(self.X_testpoints_2d[:,1]<=self.X2d_ymax))[0]
        self.X_testpoints = self.X_testpoints[idx_within_bounds]
        self.y_testpoints = self.y_testpoints[idx_within_bounds]
        self.X_testpoints_2d = self.X_testpoints_2d[idx_within_bounds]
        
    def decision_boundary_distance(self, x, grad=0):
        """Returns the distance of the given point from the decision boundary,
        i.e. the distance from the region with maximal uncertainty (0.5 
        prediction probability)"""
        return np.abs(0.5-self.classifier.predict_proba([x])[0][1])
        
    def get_decision_boundary_keypoints(self):
        """Returns the arrays of located decision boundary keypoints (both in the 
        original feature space, and in the dimensionality-reduced 2D space)
        
        Returns
        -------
        decision_boundary_points : array
            Array containing points in the original feature space which are very
            close to the true decision boundary (closer than acceptance_threshold)  
            
        decision_boundary_points_2d : array
            Array containing points in the dimensionality-reduced 2D space which 
            are very close to the true decision boundary 
        """
        if len(self.decision_boundary_points) == 0:
            raise Exception("Please call the fit method first!")
        return self.decision_boundary_points, self.decision_boundary_points_2d
        
    def _get_sorted_db_keypoint_distances(self, N=None):
        """Use a minimum spanning tree heuristic to find the N largest gaps in the 
        line constituted by the current decision boundary keypoints. 
        """
        if N == None:
            N = self.n_interpolated_keypoints
        edges = minimum_spanning_tree(squareform(pdist(self.decision_boundary_points_2d)))
        edged = np.array([euclidean(self.decision_boundary_points_2d[u], self.decision_boundary_points_2d[v]) for u,v in edges])
        gap_edge_idx = np.argsort(edged)[::-1][:N]
        edges = edges[gap_edge_idx]
        gap_distances = np.square(edged[gap_edge_idx])
        gap_probability_scores = gap_distances / np.sum(gap_distances)
        return edges, gap_distances, gap_probability_scores

    def _linear_decision_boundary_optimization(self, from_idx, to_idx, all_combinations=True, retry_neighbor_if_failed=True, step=None, suppress_output=False):
        """Use global optimization to locate the decision boundary along lines
        defined by instances from_idx and to_idx in the dataset (from_idx and to_idx
        have to contain indices from distinct classes to guarantee the existence of
        a decision boundary between them!)
        """
        step_str = ("Step "+str(step)+"/"+str(self.steps)+":") if step != None else ""
        
        retries = 4 if retry_neighbor_if_failed else 1
        for i in range(len(from_idx)):
            n = len(to_idx) if all_combinations else 1
            for j in range(n):
                from_i = from_idx[i]
                to_i = to_idx[j] if all_combinations else to_idx[i]
                for k in range(retries):
                    if k == 0:
                        fromPoint = self.Xminor[from_i]
                        toPoint = self.Xmajor[to_i]
                    else:
                        # first attempt failed, try nearest neighbors of source and destination point instead
                        _, idx = self.nn_model_2d_minorityclass.kneighbors([self.Xminor2d[from_i]])
                        fromPoint = self.Xminor[idx[0][k/2]]
                        _, idx = self.nn_model_2d_minorityclass.kneighbors([self.Xmajor2d[to_i]])
                        toPoint = self.Xmajor[idx[0][k%2]]
                    
                    if euclidean(fromPoint, toPoint) == 0:
                        break # no decision boundary between equivalent points
                
                    dbPoint = self._find_decision_boundary_along_line(fromPoint, toPoint, penalizeTangentDistance=self.penalties_enabled, penalizeExtremes=self.penalties_enabled)
                    
                    if self.decision_boundary_distance(dbPoint) <= self.acceptance_threshold:
                        dbPoint2d = self.dimensionality_reduction.transform([dbPoint])[0]
                        if dbPoint2d[0] >= self.X2d_xmin and dbPoint2d[0] <= self.X2d_xmax and dbPoint2d[1] >= self.X2d_ymin and dbPoint2d[1] <= self.X2d_ymax: 
                            self.decision_boundary_points.append(dbPoint)
                            self.decision_boundary_points_2d.append(dbPoint2d)
                            if self.verbose and not suppress_output:
                                print step_str,i*n+j,"/",len(from_idx)*n#, ": New decision boundary keypoint found using linear optimization!"
                            break
                        else:
                            if self.verbose and not suppress_output:
                                print step_str,i*n+j,"/",len(from_idx)*n, ": Rejected decision boundary keypoint (outside of plot area)"
        
    def _find_decision_boundary_along_line(self, fromPoint, toPoint, penalizeExtremes=False, penalizeTangentDistance=False):
        def objective(l, grad=0):
            # interpolate between source and destionation; calculate distance from decision boundary
            X = fromPoint + l[0] * (toPoint-fromPoint)
            error = self.decision_boundary_distance(X)
            
            if penalizeTangentDistance:
                # distance from tangent between class1 and class0 point in 2d space
                x0, y0 = self.dimensionality_reduction.transform([X])[0]
                x1, y1 = self.dimensionality_reduction.transform([fromPoint])[0]
                x2, y2 = self.dimensionality_reduction.transform([toPoint])[0]
                error += 1e-12 * np.abs((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1) / np.sqrt((y2-y1)**2+(x2-x1)**2)
            
            if penalizeExtremes:
                error += 1e-8 * np.abs(0.5-l[0]) 
            
            return error
            
        optimizer = self._get_optimizer()
        optimizer.set_min_objective(objective)
        cL = optimizer.optimize([rnd.random()])
        dbPoint = fromPoint + cL[0] * (toPoint-fromPoint)
        return dbPoint
    
    def _find_decision_boundary_on_hypersphere(self, centroid, R, penalizeKnown=False):
        def objective(phi, grad=0):
            # search on hypersphere surface in polar coordinates - map back to cartesian
            cX = centroid + polar_to_cartesian(phi, R)
            try:
                cX2d = self.dimensionality_reduction.transform([cX])[0]
                error = self.decision_boundary_distance(cX)
                if penalizeKnown:
                    # slight penalty for being too close to already known decision boundary keypoints
                    db_distances = [euclidean(cX2d, self.decision_boundary_points_2d[k]) for k in range(len(self.decision_boundary_points_2d))]
                    error += 1e-8 * ((self.mean_2d_dist - np.min(db_distances))/self.mean_2d_dist)**2
                return error
            except Exception,ex:
                print "Error in objective function:",ex
                return np.infty
            
        optimizer = self._get_optimizer(D=self.X.shape[1]-1, upper_bound=2*np.pi, iteration_budget=self.hypersphere_iteration_budget)
        optimizer.set_min_objective(objective)
        dbPhi = optimizer.optimize([rnd.random()*2*np.pi for k in range(self.X.shape[1]-1)])
        dbPoint = centroid + polar_to_cartesian(dbPhi, R)
        return dbPoint
        
    def _get_optimizer(self, D=1, upper_bound=1, iteration_budget=None):
        """Utility function creating an NLOPT optimizer with default
        parameters depending on this objects parameters
        """
        if iteration_budget == None:
            iteration_budget = self.linear_iteration_budget
        
        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, D)
        #opt.set_stopval(self.acceptance_threshold/10.0)
        opt.set_ftol_rel(1e-5)
        opt.set_maxeval(iteration_budget)
        opt.set_lower_bounds(0)
        opt.set_upper_bounds(upper_bound)
        
        return opt