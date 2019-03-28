import cv2

tensorflowNet = cv2.dnn.readNetFromTensorflow('checkpoints/hed.pb')

# Input image
img = cv2.imread('checkpoints/test.jpg')
rows, cols, channels = img.shape

# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(480, 480), swapRB=True, crop=False))

# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()
print(networkOutput)
print(networkOutput[0][0].shape)

# # Show the image with a rectagle surrounding the detected objects
cv2.imshow('Image', networkOutput[0][0])
cv2.waitKey(0)
cv2.destroyAllWindows()