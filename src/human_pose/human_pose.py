#!/usr/bin/env python3
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from neural_networks.msg import HumanPose

import mediapipe as mp


model = mp.solutions.pose.Pose(
    static_image_mode=False,
    smooth_landmarks=True,
    min_detection_confidence=0.5
)
model.__enter__()
point_names = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
]

rospy.init_node("human_pose")
result_pub = rospy.Publisher("/neural_networks/results/human_pose", HumanPose, queue_size=1)
bridge = CvBridge()


def img_callback(msg):
    if 'human_pose' not in msg.header.frame_id.split(" "):
        return

    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    h, w, c = img.shape
    results = model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pub_msg = HumanPose()
    pub_msg.header.frame_id = msg.header.frame_id
    pub_msg.point_names = point_names
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            point = Point()
            point.x = landmark.x * w
            point.y = landmark.y * h
            point.z = landmark.z
            pub_msg.points.append(point)
    else:
        pub_msg.points = [Point(-1, -1, -1) for i in range(33)]
    result_pub.publish(pub_msg)


rospy.Subscriber("/neural_networks/image", Image, callback=img_callback)
rospy.loginfo('ready')
rospy.spin()

