# Import libraries
import cv2
import numpy as np

# Import common lab functions.
from common_lab_utils import SegmentationLabGui, \
    get_sampling_rectangle, draw_sampling_rectangle, extract_training_samples


def run_segmentation_lab():
    # Set parameters.
    use_otsu = False                        # Use Otsu's method to estimate threshold automatically.
    use_adaptive_model = False              # Use adaptive method to gradually update the model continuously.
    adaptive_update_ratio = 0.1             # Update ratio for adaptive method.
    max_distance = 20                       # Maximum Mahalanobis distance we represent (in slider and uint16 image).
    initial_thresh_val = 8                  # Initial value for threshold.
    model_type = MultivariateNormalModel    # Set feature model (this is the only one available now).

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

        # TODO 1.1: Train the multivariate normal model by estimating the mean and covariance given the samples.
        self._mean = np.mean(samples, axis=0)                      
        self._covariance = np.cov(samples, rowvar=False)   

        # We are going to compute the inverse of the estimated covariance,
        # so we must ensure that the matrix is indeed invertible (not singular).
        if not np.all(self._covariance.diagonal() >= 1.e-6):
            # Regularise the covariance.
            self._covariance = self._covariance + np.identity(self._covariance.shape[0]) * 1.e-6

        # TODO 1.2: Compute the inverse of the estimated covariance.
        self._inverse_covariance = np.identity(samples.shape[1])    # Dummy solution, replace

    def compute_mahalanobis_distances(self, feature_image):
        """Computes the Mahalanobis distances for a feature image given this model"""

        samples = feature_image.reshape(-1, 3)

        # TODO 2: Compute the mahalanobis distance for each pixel feature vector wrt the multivariate normal model.
        mahalanobis = np.inf * np.ones(samples.shape[0])            # Dummy solution, replace

        return mahalanobis.reshape(feature_image.shape[:2])


def update_samples(old_samples, new_samples, update_ratio):
    """Update samples with a certain amount of new samples

    :param old_samples: The current set of samples.
    :param new_samples: A new set of samples.
    :param update_ratio: The ratio of samples to update.

    :return The updated set of samples.
    """

    # TODO 3: Implement a random update of samples given the ratio of new_samples
    old_samples = new_samples


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

    # TODO 4: Add morphological operations to reduce noise, and other fancy segmentation approaches.

    # Return updated threshold (from Otsu's) and segmented image.
    return round(thresh_scaled / scale), np.uint8(segmented_image)


def extract_features(feature_image):
    """Extracts features from the image frame

    :param feature_image: The original image frame

    :return An image of feature vectors in the np.float32 datatype
    """

    # Convert to float32.
    feature_image = np.float32(feature_image) / 255.0

    # TODO 5: Extract other/more features for each pixel.
    return feature_image


if __name__ == "__main__":
    run_segmentation_lab()
