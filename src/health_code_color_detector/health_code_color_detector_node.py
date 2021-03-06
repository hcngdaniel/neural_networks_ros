#!/usr/bin/env python3
import os
import time
import typing
import cv2
import numpy as np

import rospy
from cv_bridge import CvBridge

from neural_networks.msg import StringWithHeader
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
result_pub = rospy.Publisher("/neural_networks/results/hccd", StringWithHeader, queue_size=1)


def img_callback(msg: Image):
    if "hccd" not in msg.header.frame_id.split(' '):
        return

    cv2img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    pub_msg = StringWithHeader()
    pub_msg.header.frame_id = msg.header.frame_id

    blob = cv2.dnn.blobFromImage(
        image=cv2img,
        scalefactor=1 / 255,
        size=(640, 480),
        swapRB=False,
    )

    blob = torch.tensor(blob)
    h = model(blob)
    result = h.cpu().detach().numpy()
    result = result[0]

    class_id = np.argmax(result)
    class_name = ["red", "yellow", "green"][class_id]
    pub_msg.data = class_name

    start = time.time()
    while time.time() - start < 0.1:
        result_pub.publish(class_name)


img_sub = rospy.Subscriber("/neural_networks/image", Image, img_callback)

rospy.loginfo('ready')
rospy.spin()
