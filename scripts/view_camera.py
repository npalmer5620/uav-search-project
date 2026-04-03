#!/usr/bin/env python3
"""Display /camera/image_raw in a window using OpenCV."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.cb, 10)

    def cb(self, msg):
        dtype = np.uint8
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera', img)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
