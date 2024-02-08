# Step 2: Implement simple colour-based segmentation
First, if you haven't done so already, please read through [common_lab_utils.py](../common_lab_utils.py) and [lab_segmentation.py](../lab_segmentation.py) to get an overview of the lab.

## 1. Implement the method `_perform_training()` in `MultivariateNormalModel`
The multivariate normal distribution is characterized by a mean vector **&mu;** and a covariance matrix **&Sigma;**.

![\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) =
\frac{1}{ (2\pi)^{\frac{k}{2}} \left|\boldsymbol{\Sigma}\right|^{\frac{1}{2}}}
\exp\left[-\textstyle\frac{1}{2}(\boldsymbol{\mu} - \mathbf{x})^{T}\boldsymbol{\Sigma}^{-1}
(\boldsymbol{\mu} - \mathbf{x}) \right]](img/multivariate_normal_distribution.png)

The method `MultivariateNormalModel._performTraining()` should estimate the mean `self._mean` and the covariance `self._covariance` for the model based on the rows of training samples in the matrix `samples` collected from the sampling region.

It must also compute the inverse of the covariance matrix `self._inverse_covariance`, which we will later use to compute the Mahalanobis distance.

You can find everything you need for this in [NumPy](https://numpy.org/).
 

## 2. Implement the method `compute_mahalanobis_distances()` in `MultivariateNormalModel`
Given a multivariate normal model, the Mahalanobis distance for a vector **x** is a measure of how well the vector fits with the model.

![\mathit{d}_{\mathit{M}}(\mathbf{x}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^{T} \boldsymbol{\Sigma}^{-1} 
(\mathbf{x} - \boldsymbol{\mu})}](img/mahalanobis_distance.png)

This method should compute the Mahalanobis distance between every pixel in the input image and the estimated multivariate
normal model described by `self._mean` and `self._inverse_covariance` and return an image of Mahalanobis distances.

- Hint: For a very efficient solution, take a look at [scipy.spatial.distance.cdist].

## Experiment!
Now you should have a working segmentation method, and it is finally time to play around with it!

For example:
- Try it out on different colours/surfaces. How well does it work?
- Try changing the threshold manually using the slider.
- As we know, Otsu's method estimates a threshold between modes in a bimodal histogram distribution.
  Check out how well Otsu's method estimates a decent threshold by pressing `o`.
  When does Otsu's work well, and when is the threshold estimate bad?
  Why?


You can now continue to the [next step](3-further-work.md) to make it a bit more advanced.

[scipy.spatial.distance.cdist]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html