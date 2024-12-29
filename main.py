import cv2
import random
import numpy as np
import statistics

image = cv2.imread("input.jpeg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("image", image)
# cv2.waitKey(0)

mean = 0
variance = 10

noise = np.random.normal(mean, np.sqrt(variance), image.shape)
noisy_image = cv2.add(image.astype(np.int16), noise.astype(np.int16))
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

cv2.imshow("image_noisy", noisy_image)
cv2.waitKey(0)

cv2.imwrite("image_noisy.jpeg", noisy_image)

row, col = noisy_image.shape

window_size = 7
padding = window_size // 2
padded_image = cv2.copyMakeBorder(noisy_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

mean_filter = np.ones((window_size, window_size), np.float32) / (window_size * window_size)

mean_filtered_image = cv2.filter2D(noisy_image, -1, mean_filter, padded_image)

cv2.imshow("mean_filtered_image", mean_filtered_image)
cv2.waitKey(0)

cv2.imwrite("mean_filtered_image.jpeg", mean_filtered_image)

output_image = np.zeros(noisy_image.shape, dtype=np.uint8)
for i in range(row):
    for j in range(col):
        neighbors = padded_image[i : i + window_size, j : j + window_size]

        neighbor_mean = np.mean(neighbors)
        neighbor_variance = np.var(neighbors)

        output_v = image[i, j] - (variance / neighbor_variance) * (
            noisy_image[i, j] - neighbor_mean
        )

        if output_v < 0:
            output_v = 0
        elif output_v > 255:
            output_v = 255

        output_image[i, j] = output_v

cv2.imshow("output_image", output_image)
cv2.waitKey(0)

cv2.imwrite("output.jpeg", output_image)

cv2.destroyAllWindows()
