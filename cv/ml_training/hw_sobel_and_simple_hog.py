import numpy as np


def compute_sobel_gradients_two_loops(image):
    height, width = image.shape

    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            segment = padded_image[i - 1: i + 2, j - 1: j + 2]
            gradient_x[i - 1, j - 1] = np.sum(segment * sobel_x)
            gradient_y[i - 1, j - 1] = np.sum(segment * sobel_y)

    return gradient_x, gradient_y


def compute_gradient_magnitude(gradient_x, gradient_y):
    '''
    Compute the magnitude of the gradient given the x and y gradients.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        magnitude: numpy array of the same shape as the input [0] with the magnitude of the gradient.
    '''
    return np.sqrt(gradient_x ** 2 + gradient_y ** 2)


def compute_gradient_direction(gradient_x, gradient_y):
    '''
    Compute the direction of the gradient given the x and y gradients. Angle must be in degrees in the range (-180; 180].
    Use arctan2 function to compute the angle.

    Inputs:
        sobel_x: numpy array of the x gradient.
        sobel_y: numpy array of the y gradient.

    Returns:
        gradient_direction: numpy array of the same shape as the input [0] with the direction of the gradient.
    '''
    angle_rad = np.arctan2(gradient_y, gradient_x)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


cell_size = 7


def compute_hog(image, pixels_per_cell=(cell_size, cell_size), bins=9):
    # 1. Convert the image to grayscale if it's not already (assuming the image is in RGB or BGR)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)  # Simple averaging to convert to grayscale

    # 2. Compute gradients with Sobel filter
    gradient_x, gradient_y = compute_sobel_gradients_two_loops(image)

    # 3. Compute gradient magnitude and direction
    magnitude = compute_gradient_magnitude(gradient_x, gradient_y)
    direction = compute_gradient_direction(gradient_x, gradient_y)

    # 4. Create histograms of gradient directions for each cell
    cell_height, cell_width = pixels_per_cell
    n_cells_x = image.shape[1] // cell_width
    n_cells_y = image.shape[0] // cell_height

    histograms = np.zeros((n_cells_y, n_cells_x, bins))
    bin_edges = np.linspace(-180, 180, bins + 1)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_magnitude = magnitude[i * cell_height:(i + 1) * cell_height,
                             j * cell_width:(j + 1) * cell_width]
            cell_direction = direction[i * cell_height:(i + 1) * cell_height,
                             j * cell_width:(j + 1) * cell_width]

            hist, _ = np.histogram(cell_direction.flatten(),
                                   bins=bin_edges, weights=cell_magnitude.flatten())

            histograms[i, j] = hist

            cell_sum = np.sum(hist)
            if cell_sum > 0:
                histograms[i, j] /= cell_sum

    return histograms

