#!/usr/bin/env python3
import os
import cv2
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from neural_networks.msg import BoundingBoxes, BoundingBox

import torch


os.chdir(os.path.dirname(__file__))
os.environ['TORCH_HOME'] = '.'

rospy.init_node('yolov5')
yolo_type = rospy.get_param('yolo_type', 'yolov5l')
result_pub = rospy.Publisher('/neural_networks/results/yolov5', BoundingBoxes, queue_size=1)
bridge = CvBridge()

yolo = torch.hub.load('ultralytics/yolov5', yolo_type, pretrained=True)


def img_callback(msg):
    if 'yolov5' not in msg.header.frame_id.split(' '):
        return

    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]

    results = yolo(imgs)

    pub_msg = BoundingBoxes()
    pub_msg.header.frame_id = msg.header.frame_id
    for idx, result in results.pandas().xyxy[0].iterrows():
        box_msg = BoundingBox()
        box_msg.left = int(max(0, result['xmin']))
        box_msg.top = int(max(0, result['ymin']))
        box_msg.right = int(max(0, result['xmax']))
        box_msg.bottom = int(max(0, result['ymax']))
        box_msg.class_id = result['class']
        box_msg.class_name = result['name']
        box_msg.conf = [max(0, result['confidence'])]
        pub_msg.boxes.append(box_msg)
    result_pub.publish(pub_msg)


rospy.Subscriber('/neural_networks/image', Image, img_callback)
rospy.loginfo('ready')

rospy.spin()
