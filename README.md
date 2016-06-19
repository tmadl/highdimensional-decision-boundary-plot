Plotting high-dimensional decision boundaries
===============

An experimental, scikit-learn compatible approach to **plot high-dimensional decision boundaries**. This facilitates **intuitive understanding**, and helps improve models by allowing the **visual inspection** of misclassified regions, model complexity, and the amount of overfitting/underfitting. Unlike training/test error curves and ROC curves, this approach also visually indicates the contribution of individual instances to the result, allowing further investigation. Finally, it helps understanding which regions are likely to be misclassified, which are uncertain, and (in applications were active querying is possible) the proximity of which instances should be queried. 

The usual approach of classifying each vertex of a 2D grid to visualize decision boundaries (see e.g. the [Iris SVM example](http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html)) in two dimensions breaks down when the dimensionality is higher. Although it would in principle be possible to construct a high-dimensional grid and project it down to 2D, this is intractable in practice, since the number of grid vertices to be classified grows exponentially with the number of dimensions.

Instead, this project ...

In terms of dimensionality reduction methods, the current version supports all [matrix decomposition](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition) variants (including [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA), [Kernel PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA), [NMF](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html#sklearn.decomposition.NMF) etc.), as well as [Isomap](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap) embeddings for non-linear dimensionality reduction preserving global topology, and any other method that has an implemented and exposed `transform(X)` function. This can include supervised dimensionality reduction, such as [LDA](http://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html) (with `solver='eigen'`), which projects to the most discriminative axes.

Choosing a **dimensionality reduction method resulting in acceptabe class separation** is crucial for achieving interpretable results. 

Reliability
===============

When inspecting graphs and improving your classifier, you can trust
- The actual data points (large green and blue points)
- Misclassification feedback (red circles around the data points)
- Generated test data points colored according to your classifier predictions (tiny, faint green and blue points)

Everything else is a rough estimate intended for facilitating intuition, rather than precision; needs to be traded off against runtime; and is subject to the limitations inherent in forcing high-dimensional data into a low-dimensional plot 
- The decision boundary keypoints (large cyan squares) are guaranteed to lie very close to the decision boundary (depending on the `acceptance_threshold` parameter setting). With very small tolerance, these are fairly reliable, but do NOT provide the full picture (a complete, reliable decision boundary could only be plotted with an infinite number of keypoints). To increase reliability, decrease `acceptance_threshold` or increase the number of decision boundary keypoints
- The background shading reflects rough probability scores around the decision boundary, estimated from the generated test data points (its accuracy will depend on the number and coverage of these generated data points). As above, it is NOT a full picture (the generated data points do not provide full coverage, and only cover the space between the two classes, not beyond). To increase reliability, increase `n_generated_testpoints_per_keypoint` (or tweak the internal SVC approximating them in order to render the shading)


Usage
===============

The project requires [scikit-learn](http://scikit-learn.org/stable/install.html), [matplotlib](http://matplotlib.org/users/installing.html) and [NLopt](http://ab-initio.mit.edu/wiki/index.php/NLopt_Installation) to run.

Usage example:

```python
from uci_loader import *
```
