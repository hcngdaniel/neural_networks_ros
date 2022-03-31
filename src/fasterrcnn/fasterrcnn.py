#!/usr/bin/env python3
import os
import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
import torch
from neural_networks.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

os.environ["TORCH_HOME"] = os.path.dirname(__file__)

model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval()

bridge = CvBridge()
publisher = rospy.Publisher("/neural_networks/results/fasterrcnn", BoundingBoxes, queue_size=10)


def img_callback(msg):
    if 'fasterrcnn' not in msg.header.frame_id.split(' '):
        return

    img = bridge.imgmsg_to_cv2(msg, 'bgr8')
    imgs = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)])
    imgs = imgs / 255
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    imgs = torch.tensor(imgs, dtype=torch.float32)

    pub_msg = BoundingBoxes()

    results = model(imgs)
    for result in results:
        for i in range(len(result['boxes'])):
            box_msg = BoundingBox()
            box_msg.left = int(max(0, result['boxes'][i][0]))
            box_msg.top = int(max(0, result['boxes'][i][1]))
            box_msg.right = int(max(0, result['boxes'][i][2]))
            box_msg.bottom = int(max(0, result['boxes'][i][3]))
            box_msg.class_id = result['labels'][i]
            box_msg.class_name = COCO_CLASSES[box_msg.class_id]
            box_msg.conf = [result['scores'][i]]
            pub_msg.boxes.append(box_msg)
    pub_msg.header.seq = msg.header.seq
    pub_msg.header.frame_id = msg.header.frame_id
    publisher.publish(pub_msg)


rospy.init_node('FasterRCNN')
rospy.Subscriber('/neural_networks/image', Image, img_callback)
rospy.loginfo('ready')

rospy.spin()
