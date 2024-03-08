import numpy as np
import cv2


def draw_square(img, top_left_corner, size=800, color=(0, 0, 255), tilt_factor=0.2):
    pts = np.array(
        [
            [
                # top_left_corner[0] + size * 0.2,
                # top_left_corner[1] + size * 0.2,
                top_left_corner[0] + size * (0 + tilt_factor),
                top_left_corner[1] + size * (0 + tilt_factor),
            ],  # Top-left corner adjusted
            [
                # top_left_corner[0] + size * 0.8,
                # top_left_corner[1] + size * 0.2,
                top_left_corner[0] + size * (1 - tilt_factor),
                top_left_corner[1] + size * (0 + tilt_factor),
            ],  # Top-right corner adjusted
            [
                top_left_corner[0] + size,
                top_left_corner[1] + size,
            ],  # Bottom-right corner (same)
            [
                top_left_corner[0],
                top_left_corner[1] + size,
            ],  # Bottom-left corner (same)
        ],
        dtype=np.int32,
    )

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=5)


def draw_perspective_square_overlay(
    original_img, top_left_corner, size=800, color=(0, 0, 255), tilt_factor=0.2
):
    # Create a separate overlay image (blank with the same size as the original image)
    overlay_img = np.zeros_like(original_img)

    # Draw the square on the overlay image
    cv2.rectangle(
        overlay_img,
        top_left_corner,
        (top_left_corner[0] + size, top_left_corner[1] + size),
        color,
        -1,
    )

    # Define the points for the original square and the transformed square
    pts1 = np.float32(
        [
            [top_left_corner[0], top_left_corner[1]],
            [top_left_corner[0] + size, top_left_corner[1]],
            [top_left_corner[0], top_left_corner[1] + size],
            [top_left_corner[0] + size, top_left_corner[1] + size],
        ]
    )

    pts2 = np.float32(
        [
            [
                top_left_corner[0] + size * tilt_factor,
                top_left_corner[1] + size * tilt_factor,
            ],
            [
                top_left_corner[0] + size - size * tilt_factor,
                top_left_corner[1] + size * tilt_factor,
            ],
            [top_left_corner[0], top_left_corner[1] + size],
            [top_left_corner[0] + size, top_left_corner[1] + size],
        ]
    )

    # Get the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective transformation to the overlay image
    transformed_overlay = cv2.warpPerspective(
        overlay_img, matrix, (overlay_img.shape[1], overlay_img.shape[0])
    )

    # Overlay the transformed overlay onto the original image
    # Wherever the overlay is not black, replace the original image's pixel with the overlay's pixel
    mask = np.any(transformed_overlay != [0, 0, 0], axis=-1)
    original_img[mask] = transformed_overlay[mask]