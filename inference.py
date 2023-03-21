import numpy as np
import torch
import cv2

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# gpu
from blazeFacemark import BlazeFacemark

net = BlazeFacemark().to(gpu)
net.load_weights("blazeFacemark.pth")
img = cv2.imread("facesample3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (192, 192))
detections, score = net.predict_on_image(img)
# detections.shape
if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
detections = detections.reshape(-1, 3)
# %matplotlib inline

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(detections[:, 0], detections[:, 1], color='dodgerblue', alpha=0.8, s=2)
fig.savefig('./test.png')


def inference(img):
    net = BlazeFacemark().to(gpu)
    net.load_weights("blazeFacemark.pth")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 192))
    detections, score = net.predict_on_image(img)
    if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
    detections = detections.reshape(-1, 3)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(detections[:, 0], detections[:, 1], color='dodgerblue', alpha=0.8, s=2)
    fig.savefig('./test.png')
    img = cv2.imread("test.png")
    return img
