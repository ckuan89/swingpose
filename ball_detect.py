import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


def detect_speed(file):
    #base=os.path.basename(file)
    #filename, fileext = os.path.splitext(base)
    #try:
        video=cv2.VideoCapture(file)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_out=640
        fps = int(video.get(cv2.CAP_PROP_FPS))

        #out = cv2.VideoWriter('video_output/out_'+base ,cv2.VideoWriter_fourcc(*'XVID'), fps, (width_out, int(width_out*height/width)))
        # subtractor = cv2.createBackgroundSubtractorKNN(history=50, detectShadows=False)
        first_frame=None
        pre_frame=None
        dframe_sum=None
        roi=None
        (rx,ry,rw,rh)=(0,0,800,800)#(400,100,200,200)##

        centers_all =[]
        radius_all=[]
        final_contours = []


        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByCircularity = True
        params.minCircularity = 0.35
        params.maxCircularity = 1.1
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 500

        detector = cv2.SimpleBlobDetector_create(params)      

        i = 0
        while (video.isOpened()):
            i = i+1
            print(i)
            check, frame = video.read()
            if check == True:
                frame = cv2.resize(frame,(width_out,int(width_out*height/width)))
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                roi=gray[ry:ry+rh,rx:rx+rw]

                # mask = subtractor.apply(roi)
                if i == 1:
                    pre_frame = roi
                cv2.rectangle(frame,(rx,ry),(rx+rw,ry+rh),(255,0,0),thickness=1)
                
                d_frame = cv2.subtract(roi, pre_frame) 
                pre_frame=roi

                

                #thres_frame=cv2.erode(d_frame ,None)#,iterations=1)
                #thres_frame=cv2.dilate(thres_frame,None)#, iterations=1)
                #thres_frame = cv2.GaussianBlur(d_frame, (15, 15), 0)
                thres_frame=cv2.threshold(d_frame,20,255,cv2.THRESH_BINARY)[1]
                if i == 1:
                    dframe_sum = thres_frame
                
                # thres_frame=mask
                # (cnts,_) = cv2.findContours(thres_frame, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,offset=(rx,ry))
                # if len(cnts)>0:
                #     for contour in cnts:
                #         area = cv2.contourArea(contour)
                #         if 500 > area > 100 :
                            
                #     # c = max(cnts, key=cv2.contourArea)
                #             (centers, radius) = cv2.minEnclosingCircle(contour)
                #             cv2.circle(frame, (int(centers[0]), int(centers[1])), int(radius), (0, 0, 255), 2)
                #             cv2.circle(frame, (int(centers[0]), int(centers[1])), int(radius/100), (0, 0, 255), 2)
                # if len(cnts)>20:
                #     print(len(cnts))
                #     dframe_sum = thres_frame
                #     # centers_all.append(centers)
                #     # radius_all.append(radius)
                keypoints = detector.detect(thres_frame)
                blank = np.zeros((1, 1))  
                blobs = cv2.drawKeypoints(frame, keypoints, blank, (0, 0, 255), 
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                print(len(keypoints))
                dframe_sum=cv2.bitwise_or(dframe_sum,thres_frame)
                cv2.imshow("frame",blobs)
                cv2.imshow("dframe",dframe_sum)
                cv2.imshow("thres",thres_frame)
                #out.write(frame)
                key=cv2.waitKey(1)
                if key==ord('q'):
                    break
            else:
                break

        video.release()
        cv2.destroyAllWindows()
    # except:
    #     None

detect_speed('video/Sun-Aug--2-15-36-58_stream.mp4')