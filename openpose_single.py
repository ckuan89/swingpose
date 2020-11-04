import time
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import cv2
from random import randint
from pathlib import Path

def load_openpose(dev='cpu', mode="MPI"):
    device = dev # please change it to "gpu" if the model needs to be run on cuda.
    
    MODE = mode

    if MODE is "COCO":
        protoFile = "openpose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "openpose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    elif MODE is "MPI" :
        protoFile = "openpose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        weightsFile = "openpose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
    
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using GPU device")
    return net, nPoints, POSE_PAIRS

def pose_detect(frame,net,nPoints, POSE_PAIRS, inheight=368):



    image1 = frame
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    threshold = 0.1
 
    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = inheight
    inWidth = int((inHeight/frameHeight)*frameWidth)
    t = time.time()
    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                            (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]
    print("Time Taken = {}".format(time.time() - t))


    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            # cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            # cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 1)

    # plt.figure(figsize=[10,10])
    # plt.imshow(cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB))
    # plt.figure(figsize=[10,10])
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    print(points)
    return frame, points