#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class DepthViewer(Node):
    def __init__(self):
        super().__init__('depth_viewer')

        self.bridge = CvBridge()
        self.depth_image = None
        self.clicked_point = None

        self.subscription = self.create_subscription(
            Image,
            '/depth_cam/depth/image_raw',
            self.depth_callback,
            10
        )

        cv2.namedWindow("Depth Viewer", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Depth Viewer", self.mouse_callback)

        self.timer = self.create_timer(0.03, self.update_display)  # ~33 FPS
        self.get_logger().info('Depth viewer started.')

    def depth_callback(self, msg):
        try:
            # "passthrough" keeps original format, often uint16 for depth in mm
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_image = depth
        except Exception as e:
            self.get_logger().error(f'Failed to convert depth image: {e}')

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)

    def update_display(self):
        if self.depth_image is None:
            return

        depth = self.depth_image.copy()

        # Most depth cameras output uint16 depth in millimeters.
        # Some invalid pixels may be 0.
        valid_mask = (depth > 0)

        if np.any(valid_mask):
            valid_depth = depth[valid_mask]

            # Limit display range so the visualization is usable
            min_val = np.min(valid_depth)
            max_val = np.percentile(valid_depth, 95)

            if max_val <= min_val:
                max_val = min_val + 1

            depth_clipped = np.clip(depth, min_val, max_val)
            depth_norm = ((depth_clipped - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth, dtype=np.uint8)

        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        h, w = depth.shape[:2]

        # Show center pixel depth
        cx, cy = w // 2, h // 2
        center_depth = int(depth[cy, cx])

        cv2.circle(depth_color, (cx, cy), 4, (255, 255, 255), -1)
        cv2.putText(
            depth_color,
            f'Center: {center_depth} mm',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show clicked-point depth
        if self.clicked_point is not None:
            x, y = self.clicked_point
            if 0 <= x < w and 0 <= y < h:
                d = int(depth[y, x])
                d_cm = d / 10.0
                d_m = d / 1000.0

                cv2.circle(depth_color, (x, y), 6, (0, 255, 255), 2)
                cv2.putText(
                    depth_color,
                    f'({x},{y}) = {d} mm | {d_cm:.1f} cm | {d_m:.3f} m',
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                print(f'Clicked pixel ({x}, {y}) -> {d} mm | {d_cm:.1f} cm | {d_m:.3f} m')

        cv2.imshow("Depth Viewer", depth_color)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DepthViewer()

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