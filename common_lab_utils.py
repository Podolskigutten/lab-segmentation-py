import cv2
import numpy as np


class SegmentationLabGui:
    """A simple GUI for this lab for visualising results and choosing a threshold"""

    def __init__(self, initial_thresh_val, max_thresh_val):
        """Constructs the GUI

        :param initial_thresh_val: Initial value for the threshold.
        :param max_thresh_val: Maximum value for the threshold slider.
        """
        # Create windows.
        self.segm_win = 'Segmented frame'
        self.maha_win = 'Mahalanobis image'
        cv2.namedWindow(self.segm_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.maha_win, cv2.WINDOW_NORMAL)

        # Create slider that adjusts the threshold.
        self.slider = 'Threshold'
        thresh_setter_ref = self.__class__.thresh_val.__set__
        cv2.createTrackbar(self.slider, self.segm_win, 0, max_thresh_val, lambda val: thresh_setter_ref(self, val))

        # Set threshold (and thereby trackbar) to initial value.
        self._thresh_val = initial_thresh_val

    def __enter__(self):
        """Initialises the GUI"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroys the GUI"""
        cv2.destroyWindow(self.segm_win)
        cv2.destroyWindow(self.maha_win)

    @property
    def thresh_val(self):
        """The threshold value"""
        return self._thresh_val

    @thresh_val.setter
    def thresh_val(self, val):
        """Setter for the threshold value that also updates the slider"""
        self._thresh_val = val
        cv2.setTrackbarPos(self.slider, self.segm_win, self._thresh_val)

    def show_frame(self, frame_img):
        """Show an image in the "Segmented frame" window"""
        cv2.imshow(self.segm_win, frame_img)

    def show_mahalanobis(self, mahalanobis_img):
        """Show an image in the "Mahalanobis image" window"""
        cv2.imshow(self.maha_win, mahalanobis_img)

    def wait_key(self, time_ms):
        """Runs the highgui event loop and receives keypress events"""
        return cv2.waitKey(time_ms)


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


def draw_sampling_rectangle(image, sampling_rectangle):
    """Draw the sampling rectangle in an image

    :param image: The image to draw the rectangle in.
    :param sampling_rectangle: The sampling rectangle.
    """

    colour = (0, 0, 255)
    thickness = 3
    cv2.rectangle(image, sampling_rectangle.tl, sampling_rectangle.br, colour, thickness)


def extract_training_samples(feature_image, sampling_rectangle):
    """Extracts training samples from a sampling rectangle

    :param feature_image: An image of feature vectors.
    :param sampling_rectangle: The region in the feature image to extract samples from.

    :return The samples
    """

    patch = feature_image[sampling_rectangle.y_slice(), sampling_rectangle.x_slice()]
    samples = patch.reshape(-1, 3)
    return samples
