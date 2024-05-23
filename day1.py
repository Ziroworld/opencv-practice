import cv2

# Load an image
image = cv2.imread('test.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original image
cv2.imshow('Original Image', image)
cv2.waitKey(0)  # Wait for a key press to proceed

# Display the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)  # Wait for a key press to proceed

# Close all windows
cv2.destroyAllWindows()


