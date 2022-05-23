import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import ipdb
import skvideo.io
import torch as t


def main(model: nn.Module):
    class_labels = [  # class labels
        "sad",
        "surprise",
        "neutral",
        "happy",
        "disgust",
        "contempt",
        "anger",
        "fear",
    ]

    videodata = skvideo.io.vread("videos/video4.mp4")[::1]  # read video
    face_classifier = cv2.CascadeClassifier(  # face detector
        "dl/video/face_detector/haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)  # webcam
    # for i, frame in enumerate(videodata):
    while cap.isOpened():  # loop through frames
        ret, frame = cap.read()  # read frame
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # detect faces
        predicted_labels = []
        for (x, y, w, h) in faces:  # loop through faces
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (255, 0, 0), 2
            )  # draw rectangle
            roi_gray = gray[y : y + h, x : x + w]  # crop face
            roi_gray = cv2.resize(  # resize to 224x224
                roi_gray, (128, 128), interpolation=cv2.INTER_AREA
            )
            if np.sum([roi_gray]) != 0: # if not empty
                roi = tt.functional.to_pil_image(roi_gray) # convert to PIL image
                roi = tt.functional.to_grayscale(roi) # convert to grayscale
                roi = tt.ToTensor()(roi).unsqueeze(0) # convert to tensor

                # make a prediction on the ROI
                tensor = model(roi) # make prediction
                pred = torch.max(tensor, dim=1)[1].tolist()[0] # get prediction
                label = class_labels[pred]
                predicted_labels.append(label)
                label_position = (x, y)
                cv2.putText(  # put label on frame
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )
            else:
                cv2.putText(  # put label on frame
                    frame,
                    "No Face Found",
                    (20, 60),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )
        """
        if 'anger' in predicted_labels and not ('disgust' in predicted_labels): #(videodata.shape[0] - 1):
            import matplotlib.pyplot as plt
            plt.imshow(frame)
            plt.show()
        else:
            cv2.imshow("Emotion Detector", frame)
        """
        cv2.imshow("Emotion Detector", frame)  # show frame

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break
