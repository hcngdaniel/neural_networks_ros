#!/usr/bin/env python3
import os
import typing
import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge

from neural_networks.msg import BoundingBox, BoundingBoxes
from sensor_msgs.msg import Image

import pyzbar.pyzbar

import torch
from model import Model


device = "cuda" if torch.cuda.is_available() else "cpu"

detector = cv2.QRCodeDetector()
model = Model()
model.load_state_dict(torch.load(f'{os.path.dirname(__file__)}/model.pth', map_location=torch.device(device)))
model.eval()

rospy.init_node("hccd")
result_pub = rospy.Publisher("/neural_networks/results/hccd", BoundingBoxes, queue_size=1)


def img_callback(msg: Image):
    if 'hccd' not in msg.header.frame_id.split(' '):
        return

    global model, result_pub
    cv2_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    result_pub_msg = BoundingBoxes()
    result_pub_msg.header.frame_id = msg.header.frame_id

    decoded_text, points, _ = detector.detectAndDecode(cv2_image)
    # decoded: typing.List[pyzbar.pyzbar.Decoded] = pyzbar.pyzbar.decode(cv2_image)
    if points is None:
        result_pub.publish(result_pub_msg)
        return
    left, right = max([points[i][0][0] for i in range(len(points))]), min([points[i][0][0] for i in range(len(points))])
    top, bottom = max([points[i][0][1] for i in range(len(points))]), min([points[i][0][1] for i in range(len(points))])
    print(cv2_image)
    imgs = [cv2_image[left:right, top:bottom, :]]

    blob = cv2.dnn.blobFromImages(
        images=imgs,
        scalefactor=1 / 255,
        size=(640, 480),
        swapRB=False,
    )

    blob = torch.tensor(blob)
    h = model(blob)
    results = h.cpu().detach().numpy()

    for idx, result in enumerate(results):
        class_id = np.argmax(result)
        result_pub_msg.boxes[idx].class_id = class_id
        result_pub_msg.boxes[idx].class_name = ["red", "yellow", "green"][class_id]
        result_pub_msg.boxes[idx].conf = list(result)

    result_pub.publish(result_pub_msg)


img_sub = rospy.Subscriber("/neural_networks/image", Image, img_callback)

rospy.loginfo('ready')
rospy.spin()
