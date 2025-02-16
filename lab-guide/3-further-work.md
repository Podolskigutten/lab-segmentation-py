# Step 3. Extending the functionality
Let's make the segmentation method a bit more advanced!

## 3. Implement adaptive segmentation.
Finish the function `update_samples()`.

This function lets us update the model by replacing a fraction of the existing `old_samples_` with some `new_samples`.
This will make the model gradually change over time, and we are able to control the rate of how fast it changes by adjusting the `update_ratio` argument.
The `update_ratio` should be a number between 0 and 1, where 0.1 means that a random 10% of `old_samples` are being replaced with new one each iteration.

Keypress <kbd>a</kbd> activates/deactivates the adaptive functionality.

In its current state, this method does not perform any update at all.
Your job is to implement this method so that it works as intended.

**Suggestion 1:**  You can for instance use [numpy.random.rand] to generate a vector of random numbers between 0 and 1 with the same number of rows as the samples.
Replace columns for random numbers smaller than the `update_ratio`.

**Suggestion 2:** Another approach is to use [numpy.random.shuffle].
By first shuffling both `old_samples` and `new_samples` you can update the first N columns of `old_samples` with the first N columns of `new_samples`.
Here N should be determined based on the `update_ratio`.

How does this adaptive model work? Try adjusting the `update_ratio`.
Are the changes noticeable?

## 4. Clean up the segmentation with morphological operations
Go to the function `perform_segmentation()`.

Clean up the segmentation by using morphological operations on the binary image.
[cv::morphologyEx] can be used for this.

Connected component analysis [cv::connectedComponentsWithStats] can be used to identify the largest connected component in the binary image and remove the smaller ones.


## 5. Extract better and more features
Go to the function `extract_features()`.

Try changing the color representation of the image from **RGB** to [some other color space][cv::imgproc_color_conversions].
Does it make any difference for the segmentation?
(I like `YCrCb`).

Take a look at [cv::cvtColor].

Try using more than 3 features per pixel.
  - Measures of local uniformity
    - Local standard deviation
    - Local entropy
  - Other ideas?
  

## 6. Further experiments
Here are a few suggestions for further experiments with image segmentation.

### A. Try other models
There are many other types of models available in Python.
See for example the [scikit-learn library](https://scikit-learn.org/stable/).

Here I have used [sklearn.mixture.GaussianMixture] to create a [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) we can use instead of the multivariate normal model:

```python
from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:
    """Represents a mixture of multivariate normal models

    See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    """

    def __init__(self, samples, n_components=3, covariance_type='full'):
        """Constructs the model by training it on a set of feature samples

        :param samples: A set of feature samples
        :param n_components: The number of components in the mixture.
        :param covariance_type: Type of covariance representation, one of 'spherical', 'tied', 'diag' or 'full'.
        """

        self._gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, init_params='random')
        self._perform_training(samples)

    def _perform_training(self, samples):
        """Trains the model"""
        self._gmm.fit(samples)

        # Compute maximum likelihood for computing distances similar to Mahalanobis distances.
        num_dims = samples.shape[1]
        num_comps = self._gmm.n_components

        cov_type = self._gmm.covariance_type
        if cov_type == 'spherical':
            covariances = np.einsum('i,jk->ijk', self._gmm.covariances_, np.identity(num_dims))
        elif cov_type == 'tied':
            covariances = np.repeat(self._gmm.covariances_[np.newaxis, :, :], num_comps, axis=0)
        elif cov_type == 'diag':
            covariances = np.einsum('ij, jk->ijk', self._gmm.covariances_, np.identity(num_dims))
        elif cov_type == 'full':
            covariances = self._gmm.covariances_
        else:
            raise Exception("Unsupported covariance type")

        max_likelihood = 0
        for mean, covar, w in zip(self._gmm.means_, covariances, self._gmm.weights_):
            max_likelihood += w / np.sqrt(np.linalg.det(2 * np.pi * covar))
        self._max_log_likelihood = np.log(max_likelihood)

    def compute_mahalanobis_distances(self, image):
        """Computes the Mahalanobis distances for a feature image given this model"""

        samples = image.reshape(-1, 3)
        
        # GaussianMixture.score_samples() returns the log-likelihood, 
        # so transform this something similar to a Mahalanobis distance.
        mahalanobis = np.sqrt(2 * (self._max_log_likelihood - self._gmm.score_samples(samples)))

        return mahalanobis.reshape(image.shape[:2])
```

Add this class to the lab, and change the parameters to use this model instead:
```python
    model_type = GaussianMixtureModel    # Model: MultivariateNormalModel or GaussianMixtureModel.
```

Play around with the parameters for [sklearn.mixture.GaussianMixture].
How does this model compare to the single component normal model?


### B. Compute contours of the segmented objects and extract interesting features
In OpenCV you can use the contour of segmented areas to compute a set of different features of that object, such as center of mass, orientation, area and so on.
You can even fit ellipses or lines to your objects.

Take a look at the following OpenCV tutorials and experiment with feature extraction based on contours!
- [Contours: Getting Started]
- [Contour Features]
- [Contour Properties]


### C. Make use of your segmentation method
Here are some suggested applications:
- Insert a background image in the segmented area (like with a green screen).
- Track the pixel coordinates (center of mass) and maybe even the orientation of a coloured object.
- Detect the road in the supplied videos and estimate a line in the image you should follow to keep on the road.


[numpy.random.rand]: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
[numpy.random.shuffle]: https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
[sklearn.mixture.GaussianMixture]: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

[cv::morphologyEx]: https://docs.opencv.org/4.9.0/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f
[cv::connectedComponentsWithStats]: https://docs.opencv.org/4.9.0/d3/dc0/group__imgproc__shape.html#gae57b028a2b2ca327227c2399a9d53241
[cv::imgproc_color_conversions]: https://docs.opencv.org/4.9.0/de/d25/imgproc_color_conversions.html
[cv::cvtColor]: https://docs.opencv.org/4.9.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab

[Contours: Getting Started]: https://docs.opencv.org/4.9.0/d4/d73/tutorial_py_contours_begin.html
[Contour Features]: https://docs.opencv.org/4.9.0/dd/d49/tutorial_py_contour_features.html
[Contour Properties]: https://docs.opencv.org/4.9.0/d1/d32/tutorial_py_contour_properties.html