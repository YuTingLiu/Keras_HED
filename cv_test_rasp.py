# import the necessary packages

import imutils
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
import time
import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--write_video', help='Do you want to write the output video', default=False)
parser.add_argument('--xml', help='Path to deploy.prototxt',default='hed.xml', required=False)
parser.add_argument('--bin', help='Path to hed_pretrained_bsds.caffemodel',default='hed.bin', required=False)
parser.add_argument('--width', help='Resize input image to a specific width', default=480, type=int)
parser.add_argument('--height', help='Resize input image to a specific height', default=480, type=int)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.avi', type=str)
args = parser.parse_args()

# Load the model.
net = cv2.dnn.readNet(args.xml, args.bin)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
## Create a display window
kWinName = 'Holistically-Nested_Edge_Detection'
cv2.namedWindow(kWinName, cv2.WINDOW_AUTOSIZE)

# initialize the camera and grab a reference to the raw camera capture
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()
fps.stop()

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
while True:
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = vs.read()
    print("capture one frame : {}".format(image.shape))
    net.setInput(cv2.dnn.blobFromImage(image, size=(480, 480), swapRB=False, crop=False))
    out = net.forward()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print(out.shape)
    # out = cv2.resize(out, (image.shape[1], image.shape[0]))
    # out = 255 * out
    # out = out.astype(np.uint8)
    # out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
    # con=np.concatenate((image,out),axis=1)
    #
    cv2.imshow(kWinName,out[0][0])

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()