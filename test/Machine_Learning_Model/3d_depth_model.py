#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import threading


class JetArmDepth3DViewer(Node):

    def __init__(self):

        super().__init__('jetarm_depth_3d_viewer')
ignore image 
        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.lock = threading.Lock()

        self.depth_sub = self.create_subscription(
            Image,
            '/depth_cam/depth/image_raw',
            self.depth_callback,
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/depth_cam/depth/camera_info',
            self.camera_info_callback,
            10
        )

        self.get_logger().info('JetArm 3D depth viewer started.')


    def depth_callback(self, msg):

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            with self.lock:
                self.depth_image = depth.copy()

        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')


    def camera_info_callback(self, msg):

        with self.lock:
            self.camera_info = msg


    def get_latest_data(self):

        with self.lock:

            if self.depth_image is None or self.camera_info is None:
                return None, None

            return self.depth_image.copy(), self.camera_info


    def build_point_cloud(self, depth_image, camera_info):

        h, w = depth_image.shape[:2]

        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]

        depth_m = depth_image.astype(np.float32) / 1000.0

        valid = (depth_m > 0.05) & (depth_m < 2.0)

        v, u = np.indices((h, w))

        z = depth_m[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        points = np.stack((x, y, z), axis=-1)

        z_min = np.min(z) if len(z) > 0 else 0.0
        z_max = np.max(z) if len(z) > 0 else 1.0
        z_range = max(z_max - z_min, 1e-6)

        norm_z = (z - z_min) / z_range

        colors = np.stack((norm_z, 1.0 - norm_z, 0.5 * np.ones_like(norm_z)), axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd


def main(args=None):

    rclpy.init(args=args)

    node = JetArmDepth3DViewer()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='JetArm Interactive 3D Depth Viewer', width=1280, height=720)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0

    try:

        while rclpy.ok():

            rclpy.spin_once(node, timeout_sec=0.01)

            depth_image, camera_info = node.get_latest_data()

            if depth_image is None or camera_info is None:
                vis.poll_events()
                vis.update_renderer()
                continue

            new_pcd = node.build_point_cloud(depth_image, camera_info)

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

    except KeyboardInterrupt:
        pass

    finally:

        vis.destroy_window()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()