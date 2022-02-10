# Import libraries
import cv2
import numpy as np
import scipy.spatial.distance as ssd
from sklearn.mixture import GaussianMixture

# Import common lab functions.
from common_lab_utils import SegmentationLabGui, \
    get_sampling_rectangle, draw_sampling_rectangle, extract_training_samples


def run_segmentation_solution():
    # Set parameters.
    use_otsu = False                        # Use Otsu's method to estimate threshold automatically.
    use_adaptive_model = False              # Use adaptive method to gradually update the model continuously.
    adaptive_update_ratio = 0.1             # Update ratio for adaptive method.
    max_distance = 20                       # Maximum Mahalanobis distance we represent (in slider and uint16 image).
    initial_thresh_val = 8                  # Initial value for threshold.
    model_type = MultivariateNormalModel    # Model: MultivariateNormalModel and GaussianMixtureModel is implemented.

    # Connect to the camera.
    # Change to video file if you want to use that instead.
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read the first frame.
    success, frame = cap.read()
    if not success:
        return

    # Construct sampling region based on image dimensions.
    sampling_rectangle = get_sampling_rectangle(frame.shape)

    # Train first model based on samples from the first image.
    feature_image = extract_features(frame)
    samples = extract_training_samples(feature_image, sampling_rectangle)
    model = model_type(samples)

    # Set up a simple gui for the lab (based on OpenCV highgui) and run the main loop.
    with SegmentationLabGui(initial_thresh_val, max_distance) as gui:
        while True:
            # Read next frame.
            success, frame = cap.read()
            if not success:
                break

            # Extract features.
            feature_image = extract_features(frame)

            # Update if using adaptive model
            if use_adaptive_model:
                new_samples = extract_training_samples(feature_image, sampling_rectangle)
                update_samples(samples, new_samples, adaptive_update_ratio)
                model = model_type(samples)

            # Compute how well the pixel features fit with the model.
            mahalanobis_img = model.compute_mahalanobis_distances(feature_image)

            # Segment out the areas of the image that fits well enough.
            gui.thresh_val, segmented = perform_segmentation(mahalanobis_img, gui.thresh_val, use_otsu, max_distance)

            # Highlight the segmented area in green in the input frame and draw the sampling rectangle.
            frame[segmented > 0] *= np.uint8([0, 1, 0])
            draw_sampling_rectangle(frame, sampling_rectangle)

            # Normalise the Mahalanobis image so that it represents [0, max_distance] in visualisation.
            mahalanobis_img = mahalanobis_img / max_distance

            # Show the results
            gui.show_frame(frame)
            gui.show_mahalanobis(mahalanobis_img)

            # Update the GUI and wait a short time for input from the keyboard.
            key = gui.wait_key(1)

            # React to keyboard commands.
            if key == ord('q'):
                print("Quitting")
                break

            elif key == ord(' '):
                print("Extracting samples manually")
                samples = extract_training_samples(feature_image, sampling_rectangle)
                model = model_type(samples)

            elif key == ord('o'):
                use_otsu = not use_otsu
                print(f"Use Otsu's: {use_otsu}")

            elif key == ord('a'):
                use_adaptive_model = not use_adaptive_model
                print(f"Use adaptive model: {use_adaptive_model}")

    # Stop video source.
    cap.release()


class MultivariateNormalModel:
    """Represents a multivariate normal model"""

    def __init__(self, samples):
        """Constructs the model by training it on a set of feature samples

        :param samples: A set of feature samples
        """

        self._perform_training(samples)

    def _perform_training(self, samples):
        """Trains the model"""

        self._mean = np.mean(samples, axis=0)
        self._covariance = np.cov(samples, rowvar=False)
        self._inverse_covariance = np.linalg.inv(self._covariance)

    def compute_mahalanobis_distances(self, feature_image):
        """Computes the Mahalanobis distances for a feature image given this model"""

        samples = feature_image.reshape(-1, 3)
        mahalanobis = ssd.cdist(samples, self._mean[np.newaxis, :], metric='mahalanobis', VI=self._inverse_covariance)

        return mahalanobis.reshape(feature_image.shape[:2])


def update_samples(old_samples, new_samples, update_ratio):
    """Update samples with a certain amount of new samples

    :param old_samples: The current set of samples.
    :param new_samples: A new set of samples.
    :param update_ratio: The ratio of samples to update.

    :return The updated set of samples.
    """

    rand_num = np.random.rand(new_samples.shape[0])
    selected_samples = rand_num < update_ratio
    old_samples[selected_samples] = new_samples[selected_samples]


def perform_segmentation(distance_image, thresh, use_otsu, max_dist_value):
    """Segment the distance image by thresholding

    :param distance_image: An image of "signature distances".
    :param thresh: Threshold value.
    :param use_otsu: Set to True to use Otsu's method to estimate the threshold value.
    :param max_dist_value: The maximum distance value to represent after rescaling.

    :return The updated threshold value and segmented image
    """

    # We need to represent the distances in uint16 because of OpenCV's implementation of Otsu's method.
    scale = np.iinfo(np.uint16).max / max_dist_value
    distances_scaled = np.uint16(np.clip(distance_image * scale, 0, np.iinfo(np.uint16).max))
    thresh_scaled = thresh * scale

    # Perform thresholding
    thresh_type = cv2.THRESH_BINARY_INV
    if use_otsu:
        thresh_type |= cv2.THRESH_OTSU
    thresh_scaled, segmented_image = cv2.threshold(distances_scaled, thresh_scaled, 255, thresh_type)

    # Perform cleanup using morphological operations.
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, structuring_element)
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, structuring_element)

    # Return updated threshold (from Otsu's) and segmented image.
    return round(thresh_scaled / scale), np.uint8(segmented_image)


def extract_features(feature_image):
    """Extracts features from the image frame

    :param feature_image: The original image frame

    :return An image of feature vectors in the np.float32 datatype
    """

    # Convert to float32.
    feature_image = np.float32(feature_image) / 255.0

    # Choose a colour format:
    # return feature_image
    # return cv2.cvtColor(feature_image, cv2.COLOR_BGR2HSV)
    # return cv2.cvtColor(feature_image, cv2.COLOR_BGR2HLS)
    # return cv2.cvtColor(feature_image, cv2.COLOR_BGR2Lab)
    # return cv2.cvtColor(feature_image, cv2.COLOR_BGR2Luv)
    # return cv2.cvtColor(feature_image, cv2.COLOR_BGR2XYZ)
    return cv2.cvtColor(feature_image, cv2.COLOR_BGR2YCrCb)


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


if __name__ == "__main__":
    run_segmentation_solution()
