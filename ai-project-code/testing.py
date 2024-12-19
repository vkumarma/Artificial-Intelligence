
import cv2
import numpy as np

def draw_polygon(image, center, num_sides, radius, color):

    # Unpack center coordinates
    center_x, center_y = center
    height, width, _ = image.shape
    # Generate the vertices of the polygon
    vertices = []
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides  # Angle in radians
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        vertices.append([x, y])

    # Convert polygon to a NumPy array for easier manipulation
    vertices = np.array(vertices)

    # Calculate the minimum x and y values
    min_x = min(vertices[:, 0])
    min_y = min(vertices[:, 1])

    # Shift the polygon only if it goes out of bounds
    if min_x < 0:
        vertices[:, 0] -= min_x
    if min_y < 0:
        vertices[:, 1] -= min_y

    # Ensure the polygon doesn't exceed the maximum bounds
    max_x = max(vertices[:, 0])
    max_y = max(vertices[:, 1])

    if max_x > width:
        vertices[:, 0] -= (max_x - width)
    if max_y > height:
        vertices[:, 1] -= (max_y - height)

    vertices = vertices.tolist()
    print(vertices)

    # Convert vertices to a NumPy array and reshape for OpenCV

    polygon = np.array(vertices, np.int32).reshape((-1, 1, 2))

    # Fill the polygon
    cv2.fillPoly(image, [polygon], color)

    return image


# Define parameters for the polygon
center = (480,480)  # Center of the polygon
num_sides = 6        # Number of sides (e.g., 6 for a hexagon)
radius = 100         # Radius of the polygon
color = (0, 255, 0)  # Green color in BGR
image = np.full(shape=(500,500,3), fill_value=255, dtype=np.uint8)
# Draw the polygon
image_with_polygon = draw_polygon(image, center, num_sides, radius, color)

# Display the image
cv2.imshow("Polygon", image_with_polygon)
cv2.waitKey(0)
cv2.destroyAllWindows()