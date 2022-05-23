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
    class_labels = [
        "sad",
        "surprise",
        "neutral",
        "happy",
        "disgust",
        "contempt",
        "anger",
        "fear",
    ]

    videodata = skvideo.io.vread("videos/video4.mp4")[::1]
    face_classifier = cv2.CascadeClassifier(
        "dl/video/face_detector/haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    # for frame in videodata:
    while cap.isOpened():
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (128, 128), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = tt.functional.to_pil_image(roi_gray)
                roi = tt.functional.to_grayscale(roi)
                roi = tt.ToTensor()(roi).unsqueeze(0)

                # make a prediction on the ROI
                tensor = model(roi)
                pred = torch.max(tensor, dim=1)[1].tolist()
                label = class_labels[pred[0]]

                label_position = (x, y)
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )
            else:
                cv2.putText(
                    frame,
                    "No Face Found",
                    (20, 60),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
