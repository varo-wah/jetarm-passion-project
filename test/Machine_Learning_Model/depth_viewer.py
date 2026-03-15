#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class DepthViewer(Node):
    def __init__(self):
        super().__init__('depth_viewer')

        self.bridge = CvBridge()
        self.depth_image_mm = None
        self.clicked_point = None

        self.depth_topic = '/camera/depth/image_raw'
        self.camera_info_topic = '/camera/depth/camera_info'

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.logged_intrinsics_fallback = False

        self.depth_window_name = 'Depth Viewer'
        self.pointcloud_window_name = 'Depth Viewer 3D'
        self.point_stride = 4
        self.max_points = 18000
        self.max_depth_m = 4.0
        self.viewer_width = 1280
        self.viewer_height = 720

        self.yaw = np.deg2rad(35.0)
        self.pitch = np.deg2rad(-20.0)
        self.zoom = 1.2
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.rotating = False
        self.panning = False
        self.last_mouse_pos = None

        self.depth_subscription = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        cv2.namedWindow(self.depth_window_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.pointcloud_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.depth_window_name, 960, 540)
        cv2.resizeWindow(self.pointcloud_window_name, self.viewer_width, self.viewer_height)
        cv2.setMouseCallback(self.depth_window_name, self.depth_mouse_callback)
        cv2.setMouseCallback(self.pointcloud_window_name, self.pointcloud_mouse_callback)

        self.timer = self.create_timer(0.05, self.update_display)
        self.get_logger().info(
            'Depth viewer started. Opening a 2D depth view and a live 3D point-cloud window.'
        )

    def camera_info_callback(self, msg):
        if len(msg.k) < 9 or msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            return

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.get_logger().info(
            f'Using camera intrinsics from {self.camera_info_topic}: '
            f'fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}'
        )
        self.destroy_subscription(self.camera_info_subscription)
        self.camera_info_subscription = None

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert depth image: {exc}')
            return

        if depth is None:
            return

        if np.issubdtype(depth.dtype, np.floating):
            depth_m = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        else:
            depth_mm = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)

        self.depth_image_mm = depth_mm

    def depth_mouse_callback(self, event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)

    def pointcloud_mouse_callback(self, event, x, y, flags, param):
        del param

        if event == cv2.EVENT_LBUTTONDOWN:
            self.rotating = True
            self.last_mouse_pos = (x, y)
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.panning = True
            self.last_mouse_pos = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.rotating = False
            self.last_mouse_pos = None
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.panning = False
            self.last_mouse_pos = None
            return

        if event == cv2.EVENT_MOUSEMOVE and self.last_mouse_pos is not None:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]

            if self.rotating:
                self.yaw += dx * 0.01
                self.pitch = np.clip(self.pitch + dy * 0.01, -1.4, 1.4)
            elif self.panning:
                self.pan_x += dx * 0.002
                self.pan_y -= dy * 0.002

            self.last_mouse_pos = (x, y)
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            self.adjust_zoom(flags)

    def adjust_zoom(self, flags):
        if hasattr(cv2, 'getMouseWheelDelta'):
            delta = cv2.getMouseWheelDelta(flags)
        else:
            delta = 1 if flags > 0 else -1

        if delta > 0:
            self.zoom = min(self.zoom * 1.1, 5.0)
        elif delta < 0:
            self.zoom = max(self.zoom / 1.1, 0.3)

    def get_intrinsics(self, width, height):
        if all(value is not None for value in (self.fx, self.fy, self.cx, self.cy)):
            return self.fx, self.fy, self.cx, self.cy

        if not self.logged_intrinsics_fallback:
            self.get_logger().warn(
                f'No camera info received on {self.camera_info_topic}; '
                'using an estimated focal length for the 3D preview.'
            )
            self.logged_intrinsics_fallback = True

        fx = float(max(width, height) * 1.1)
        fy = fx
        cx = float(width) / 2.0
        cy = float(height) / 2.0
        return fx, fy, cx, cy

    def create_depth_preview(self, depth_mm):
        valid_mask = (depth_mm > 0)

        if np.any(valid_mask):
            valid_depth = depth_mm[valid_mask]
            min_val = float(np.min(valid_depth))
            max_val = float(np.percentile(valid_depth, 95))

            if max_val <= min_val:
                max_val = min_val + 1.0

            depth_clipped = np.clip(depth_mm.astype(np.float32), min_val, max_val)
            depth_norm = ((depth_clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            depth_norm[~valid_mask] = 0
        else:
            depth_norm = np.zeros_like(depth_mm, dtype=np.uint8)

        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)

    def create_point_cloud(self, depth_mm):
        height, width = depth_mm.shape
        fx, fy, cx, cy = self.get_intrinsics(width, height)

        sampled_depth = depth_mm[::self.point_stride, ::self.point_stride].astype(np.float32)
        valid_mask = (sampled_depth > 0.0) & (sampled_depth <= self.max_depth_m * 1000.0)

        if not np.any(valid_mask):
            return None, None, None

        v_coords, u_coords = np.indices(sampled_depth.shape, dtype=np.float32)
        u_coords *= self.point_stride
        v_coords *= self.point_stride

        depth_m = sampled_depth / 1000.0
        x_coords = (u_coords - cx) * depth_m / fx
        y_coords = (v_coords - cy) * depth_m / fy

        points = np.stack(
            (
                x_coords[valid_mask],
                -y_coords[valid_mask],
                depth_m[valid_mask],
            ),
            axis=1
        ).astype(np.float32)
        depths = depth_m[valid_mask].astype(np.float32)

        if points.shape[0] > self.max_points:
            step = int(np.ceil(points.shape[0] / self.max_points))
            points = points[::step]
            depths = depths[::step]

        colors = self.depths_to_colors(depths)
        return points, colors, depths

    def depths_to_colors(self, depths):
        near = float(np.min(depths))
        far = float(np.percentile(depths, 95))
        if far <= near:
            far = near + 0.001

        normalized = np.clip((depths - near) / (far - near), 0.0, 1.0)
        color_values = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
        color_map_input = color_values.reshape(-1, 1)
        return cv2.applyColorMap(color_map_input, cv2.COLORMAP_TURBO).reshape(-1, 3)

    def rotation_matrix(self):
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        cos_pitch = np.cos(self.pitch)
        sin_pitch = np.sin(self.pitch)

        yaw_matrix = np.array(
            [
                [cos_yaw, 0.0, sin_yaw],
                [0.0, 1.0, 0.0],
                [-sin_yaw, 0.0, cos_yaw],
            ],
            dtype=np.float32
        )
        pitch_matrix = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_pitch, -sin_pitch],
                [0.0, sin_pitch, cos_pitch],
            ],
            dtype=np.float32
        )
        return pitch_matrix @ yaw_matrix

    def render_point_cloud(self, points, colors, depths):
        canvas = np.full((self.viewer_height, self.viewer_width, 3), 28, dtype=np.uint8)

        if points is None or colors is None or depths is None:
            cv2.putText(
                canvas,
                'Waiting for valid depth data...',
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (220, 220, 220),
                2,
                cv2.LINE_AA
            )
            return canvas

        center = np.median(points, axis=0).astype(np.float32)
        centered_points = points - center

        rotation = self.rotation_matrix()
        rotated_points = centered_points @ rotation.T

        scene_radius = float(np.percentile(np.linalg.norm(centered_points, axis=1), 90))
        scene_radius = max(scene_radius, 0.25)

        rotated_points[:, 0] += self.pan_x * scene_radius
        rotated_points[:, 1] += self.pan_y * scene_radius
        rotated_points[:, 2] += (scene_radius * 3.0) / self.zoom

        valid_mask = rotated_points[:, 2] > 0.05
        if not np.any(valid_mask):
            return canvas

        rotated_points = rotated_points[valid_mask]
        colors = colors[valid_mask]
        depths = depths[valid_mask]

        draw_order = np.argsort(rotated_points[:, 2])[::-1]
        rotated_points = rotated_points[draw_order]
        colors = colors[draw_order]
        depths = depths[draw_order]

        focal = min(self.viewer_width, self.viewer_height) * 0.95
        projected_x = (rotated_points[:, 0] / rotated_points[:, 2] * focal + self.viewer_width / 2.0)
        projected_y = (-rotated_points[:, 1] / rotated_points[:, 2] * focal + self.viewer_height / 2.0)

        in_bounds = (
            (projected_x >= 0.0) & (projected_x < self.viewer_width) &
            (projected_y >= 0.0) & (projected_y < self.viewer_height)
        )

        projected_x = projected_x[in_bounds].astype(np.int32)
        projected_y = projected_y[in_bounds].astype(np.int32)
        colors = colors[in_bounds]
        depths = depths[in_bounds]
        visible_points = rotated_points[in_bounds]

        if projected_x.size == 0:
            return canvas

        point_sizes = np.clip((1.8 / visible_points[:, 2]) * 4.0, 1.0, 5.0).astype(np.int32)

        for x_coord, y_coord, color, radius in zip(projected_x, projected_y, colors, point_sizes):
            cv2.circle(
                canvas,
                (int(x_coord), int(y_coord)),
                int(radius),
                tuple(int(channel) for channel in color),
                -1,
                lineType=cv2.LINE_AA
            )

        self.draw_overlay(canvas, depths)
        return canvas

    def draw_overlay(self, canvas, depths):
        text_color = (235, 235, 235)
        near_depth = float(np.min(depths))
        far_depth = float(np.percentile(depths, 95))

        cv2.putText(
            canvas,
            'Live 3D Depth Viewer',
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f'Depth range: {near_depth:.2f}m to {far_depth:.2f}m',
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            text_color,
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            'Drag left mouse to rotate | Drag right mouse to pan | +/- or wheel to zoom | R to reset | Q to quit',
            (25, self.viewer_height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            1,
            cv2.LINE_AA
        )

        legend = np.linspace(255, 0, 256, dtype=np.uint8).reshape(-1, 1)
        legend = cv2.applyColorMap(legend, cv2.COLORMAP_TURBO)
        legend = cv2.resize(legend, (24, 180), interpolation=cv2.INTER_LINEAR)
        top = 100
        left = self.viewer_width - 70
        canvas[top:top + 180, left:left + 24] = legend

        cv2.putText(
            canvas,
            f'{near_depth:.2f}m',
            (left - 5, top + 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f'{far_depth:.2f}m',
            (left - 5, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA
        )

    def reset_view(self):
        self.yaw = np.deg2rad(35.0)
        self.pitch = np.deg2rad(-20.0)
        self.zoom = 1.2
        self.pan_x = 0.0
        self.pan_y = 0.0

    def handle_keyboard(self):
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            self.get_logger().info('Shutting down depth viewer.')
            cv2.destroyAllWindows()
            rclpy.shutdown()
            return

        if key in (ord('+'), ord('=')):
            self.zoom = min(self.zoom * 1.1, 5.0)
        elif key in (ord('-'), ord('_')):
            self.zoom = max(self.zoom / 1.1, 0.3)
        elif key in (ord('r'), ord('R')):
            self.reset_view()
        elif key in (ord('a'), ord('A')):
            self.yaw -= 0.08
        elif key in (ord('d'), ord('D')):
            self.yaw += 0.08
        elif key in (ord('w'), ord('W')):
            self.pitch = max(self.pitch - 0.08, -1.4)
        elif key in (ord('s'), ord('S')):
            self.pitch = min(self.pitch + 0.08, 1.4)

    def update_display(self):
        if self.depth_image_mm is None:
            self.handle_keyboard()
            return

        depth_mm = self.depth_image_mm.copy()
        depth_preview = self.create_depth_preview(depth_mm)

        height, width = depth_mm.shape
        center_x, center_y = width // 2, height // 2
        center_depth_mm = int(depth_mm[center_y, center_x])

        cv2.circle(depth_preview, (center_x, center_y), 4, (255, 255, 255), -1)
        cv2.putText(
            depth_preview,
            f'Center: {center_depth_mm} mm',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if self.clicked_point is not None:
            x_coord, y_coord = self.clicked_point
            if 0 <= x_coord < width and 0 <= y_coord < height:
                clicked_depth_mm = int(depth_mm[y_coord, x_coord])
                clicked_depth_cm = clicked_depth_mm / 10.0
                clicked_depth_m = clicked_depth_mm / 1000.0

                cv2.circle(depth_preview, (x_coord, y_coord), 6, (0, 255, 255), 2)
                cv2.putText(
                    depth_preview,
                    (
                        f'({x_coord},{y_coord}) = {clicked_depth_mm} mm | '
                        f'{clicked_depth_cm:.1f} cm | {clicked_depth_m:.3f} m'
                    ),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        points, colors, depths = self.create_point_cloud(depth_mm)
        pointcloud_view = self.render_point_cloud(points, colors, depths)

        cv2.imshow(self.depth_window_name, depth_preview)
        cv2.imshow(self.pointcloud_window_name, pointcloud_view)
        self.handle_keyboard()


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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
