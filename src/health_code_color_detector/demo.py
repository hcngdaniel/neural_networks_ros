#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from neural_networks.msg import BoundingBoxes


def result_callback(msg: BoundingBoxes):
    global boxes, result_frame_id
    boxes = msg.boxes
    result_frame_id = msg.header.frame_id


rospy.init_node("hccd_test")
img_pub = rospy.Publisher("/neural_networks/image", Image, queue_size=10)
result_sub = rospy.Subscriber("/neural_networks/results/hccd", BoundingBoxes, callback=result_callback)

boxes = []
result_frame_id = None

cap = cv2.VideoCapture()
while cap.isOpened():
    ret, frame = cap.read()

    imgmsg = CvBridge().cv2_to_imgmsg(frame, "bgr8")
    expected_frame_id = imgmsg.header.frame_id
    img_pub.publish(imgmsg)

    while result_frame_id != expected_frame_id:
        rospy.Rate(20).sleep()

    result_frame_id = None
    for box in boxes:
        color = [(0, 0, 255), (0, 255, 255), (0, 255, 0)][box.class_id]
        cv2.rectangle(frame, (box.left, box.top), (box.right, box.bottom), color, 3)
        cv2.putText(frame, box.class_name, (box.left, box.top), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(16)
    if key in [27, ord('q')]:
        break
cap.release()
cv2.destroyAllWindows()
