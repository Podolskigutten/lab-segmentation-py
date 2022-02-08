import cv2
import numpy as np
import scipy.spatial.distance as ssd
from sklearn.mixture import GaussianMixture


class Rectangle:
    """Represents a geometric rectangle"""

    def __init__(self, top_left, bottom_right):
        """Constructs a rectangle.

        :param top_left: A tuple representing the top left point (x1, y1) in the rectangle.
        :param bottom_right: A tuple representing the bottom right point (x2, y2) in the rectangle.
        """
        self._top_left = top_left
        self._bottom_right = bottom_right

    @property
    def tl(self):
        """The top left point of the rectangle"""
        return self._top_left

    @property
    def br(self):
        """The bottom right point of the rectangle"""
        return self._bottom_right

    def x_slice(self):
        """Extract a slice object for the x-range of the rectangle"""
        return slice(self.tl[0], self.br[0])

    def y_slice(self):
        """Extract a slice object for the y-range of the rectangle"""
        return slice(self.tl[1], self.br[1])


def get_sampling_rectangle(img_shape, rect_size=(80, 100)):
    """Computes the sampling rectangle based on the image and rectangle sizes

    :param img_shape: The shape of the images, as returned by numpy.ndarray.shape.
    :param rect_size: The size of the sampling rectangle given as the tuple (height, width)

    :return A Rectangle representing the sampling rectangle
    """

    img_height, img_width, _ = img_shape
    rect_height, rect_width = rect_size

    center_x = img_width // 2
    center_y = (img_height * 4) // 5
    x_left = np.clip(center_x - rect_width // 2, 0, img_width)
    x_right = np.clip(x_left + rect_width, 0, img_width)
    y_top = np.clip(center_y - rect_height // 2, 0, img_height)
    y_bottom = np.clip(y_top + rect_height, 0, img_height)

    return Rectangle((x_left, y_top), (x_right, y_bottom))


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


def extract_training_samples(feature_image, sampling_rectangle):
    """Extracts training samples from a sampling rectangle

    :param feature_image: An image of feature vectors.
    :param sampling_rectangle: The region in the feature image to extract samples from.

    :return The samples
    """

    patch = feature_image[sampling_rectangle.y_slice(), sampling_rectangle.x_slice()]
    samples = patch.reshape(-1, 3)
    return samples


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


def draw_sampling_rectangle(image, sampling_rectangle):
    """Draw the sampling rectangle in an image

    :param image: The image to draw the rectangle in.
    :param sampling_rectangle: The sampling rectangle.
    """

    colour = (0, 0, 255)
    thickness = 3
    cv2.rectangle(image, sampling_rectangle.tl, sampling_rectangle.br, colour, thickness)


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


class MultivariateNormalModel:
    """Represents a multivariate normal model"""

    def __init__(self, samples):
        """Constructs the model by training it on a set of feature samples

        :param samples: A set of feature samples
        """

        self._perform_training(samples)

    def _perform_training(self, samples):
        """Trains the model"""
        self.mean = np.mean(samples, axis=0)
        self.covariance = np.cov(samples, rowvar=False)
        self.inverse_covariance = np.linalg.inv(self.covariance)

    def compute_mahalanobis_distances(self, feature_image):
        """Computes the Mahalanobis distances for a feature image given this model"""
        samples = feature_image.reshape(-1, 3)
        mahalanobis = ssd.cdist(samples, self.mean[None, :], metric='mahalanobis', VI=self.inverse_covariance)

        return mahalanobis.reshape(feature_image.shape[:2])


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

        # At least, compute something similar to Mahalanobis distances for this model type.
        samples = image.reshape(-1, 3)
        mahalanobis = np.sqrt(2 * (self._max_log_likelihood - self._gmm.score_samples(samples)))

        return mahalanobis.reshape(image.shape[:2])


def run_segmentation_lab():
    # Set parameters.
    use_otsu = False                        # Use Otsu's method to estimate threshold automatically.
    use_adaptive_model = False              # Use adaptive method to gradually update the model continuously.
    adaptive_update_ratio = 0.1             # Update ratio for adaptive method.
    max_std_dev = 20                        # Maximum Mahalanobis distance we represesent (in slider and uint16 image).
    thresh_val = 8                          # Default value for threshold.
    model_type = MultivariateNormalModel    # Model: MultivariateNormalModel and GaussianMixtureModel is implemented.

    # Create windows and gui components.
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mahalanobis_img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('segmented', cv2.WINDOW_NORMAL)

    def on_change(val):
        """Callback for trackbar"""
        nonlocal thresh_val
        thresh_val = val

    cv2.createTrackbar('Threshold', 'frame', thresh_val, max_std_dev, on_change)

    # Connect to the camera.
    # Change to video file if you want to use that instead.
    video_source = 0
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read first frame.
    success, frame = cap.read()
    if not success:
        return

    # Construct sampling region based on image dimensions.
    sampling_rectangle = get_sampling_rectangle(frame.shape)

    # Train first model based on samples from the first image.
    feature_image = extract_features(frame)
    samples = extract_training_samples(feature_image, sampling_rectangle)
    model = model_type(samples)

    # The main loop in the program.
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
        thresh_val, segmented = perform_segmentation(mahalanobis_img, thresh_val, use_otsu, max_std_dev)

        # Set segmented area to green.
        frame[segmented > 0] = (0, 255, 0)

        # Draw current frame.
        draw_sampling_rectangle(frame, sampling_rectangle)
        viz = (mahalanobis_img - mahalanobis_img.min()) / (mahalanobis_img.max() - mahalanobis_img.min())

        cv2.imshow('frame', frame)
        cv2.imshow('mahalanobis_img', viz)
        cv2.imshow('segmented', segmented)
        cv2.setTrackbarPos('Threshold', 'frame', thresh_val)

        # Update the GUI and wait a short time for input from the keyboard.
        key = cv2.waitKey(1) & 0xFF

        # React to commands from the keyboard.
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_segmentation_lab()
