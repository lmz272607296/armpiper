#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import cv_bridge
import image_geometry
import sys
import time
import os
import json
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics 未安装。请运行: pip install ultralytics")
    sys.exit(1)

BOTTLE = 'bottle'
FRUIT = 'fruit'


def normalize_object_keyword(value):
    normalized = str(value).strip().lower()
    aliases = {
        BOTTLE: BOTTLE,
        FRUIT: FRUIT,
        'furit': FRUIT,
        'cube': FRUIT,
        'orange': FRUIT,
        'apple': FRUIT,
    }
    return aliases.get(normalized, normalized)

class UltimateYoloNodeCPU(Node):
    def __init__(self):
        super().__init__('ultimate_yolo_node_cpu')
        
        self.declare_parameter('model_path', os.path.expanduser('~/yolo11n.pt'))
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('input_topic_color', '/camera/camera/color/image_raw')
        # 👇 注意这里！必须改成 aligned_depth_to_color
        self.declare_parameter('input_topic_depth', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('input_topic_camera_info', '/camera/camera/color/camera_info')
        self.declare_parameter('output_topic', '/yolo_3d_detections_json')
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('max_det', 30)
        self.declare_parameter('enable_display', True)
        self.declare_parameter('log_interval_sec', 2.0)

        model_path = self.get_parameter('model_path').value
        self.conf_thres = self.get_parameter('conf_threshold').value
        self.nms_thres = self.get_parameter('nms_threshold').value
        input_topic_color = self.get_parameter('input_topic_color').value
        input_topic_depth = self.get_parameter('input_topic_depth').value
        input_topic_camera_info = self.get_parameter('input_topic_camera_info').value
        output_topic = self.get_parameter('output_topic').value
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.max_det = int(self.get_parameter('max_det').value)
        self.enable_display = bool(self.get_parameter('enable_display').value)
        self.log_interval_sec = float(self.get_parameter('log_interval_sec').value)
        self.last_log_time = 0.0
        
        try:
            self.model = YOLO(model_path)
            self.get_logger().info(f"YOLO PyTorch 模型已成功从 {model_path} 加载")
            # 预热模型
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model(dummy_img, imgsz=self.imgsz, max_det=self.max_det, verbose=False)
            self.get_logger().info("YOLO 模型已预热。")
        except Exception as e:
            self.get_logger().fatal(f"加载 YOLO 模型失败: {e}")
            raise e

        self.bridge = cv_bridge.CvBridge()
        self.camera_model = None
        
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        self.color_sub = message_filters.Subscriber(self, Image, input_topic_color, qos_profile=qos_profile)
        self.depth_sub = message_filters.Subscriber(self, Image, input_topic_depth, qos_profile=qos_profile)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.sync_callback)
           
        self.camera_info_sub = self.create_subscription(
            CameraInfo, input_topic_camera_info, self.camera_info_callback, qos_profile=qos_profile)
           
        self.detection_pub = self.create_publisher(String, output_topic, 10)
        
        self.get_logger().info(
            f"YOLO 节点已初始化。发布 JSON 结果到: {output_topic}，"
            f"imgsz={self.imgsz}, max_det={self.max_det}, enable_display={self.enable_display}"
        )

    def camera_info_callback(self, msg):
        if self.camera_model is None:
            self.camera_model = image_geometry.PinholeCameraModel()
            self.camera_model.fromCameraInfo(msg)
            self.get_logger().info("相机内参模型已初始化")
            self.destroy_subscription(self.camera_info_sub)

    def sync_callback(self, color_msg: Image, depth_msg: Image):
        if self.camera_model is None:
            self.get_logger().warn("相机内参模型尚未就绪，跳过此帧")
            return
           
        try:
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f"图像转换错误: {e}")
            return
           
        results = self.model(
            cv_color_image,
            conf=self.conf_thres,
            iou=self.nms_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
        )
        result = results[0]

        detections_list = []
        display_color_img = cv_color_image.copy() if self.enable_display else None
        
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = result.names[class_id]
            score = float(box.conf)
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            detection = {
                "class_name": class_name,
                "object_type": normalize_object_keyword(class_name),
                "instance_name": class_name,
                "position_label": None,
                "position_rank": None,
                "confidence": score,
                "box": [x1, y1, x2, y2],
                "center_pixel": [center_x, center_y],
                "center_3d": None
            }
             
            if 0 <= center_y < cv_depth_image.shape[0] and 0 <= center_x < cv_depth_image.shape[1]:
                patch_size = 5
                half_patch = patch_size // 2
                y_start = max(0, center_y - half_patch)
                y_end = min(cv_depth_image.shape[0], center_y + half_patch + 1)
                x_start = max(0, center_x - half_patch)
                x_end = min(cv_depth_image.shape[1], center_x + half_patch + 1)
                
                depth_patch = cv_depth_image[y_start:y_end, x_start:x_end]
                valid_depths = depth_patch[depth_patch > 0]

                if len(valid_depths) > 0:
                    depth_m = self.depth_to_meters(float(np.median(valid_depths)), depth_msg.encoding)
                    if 0.1 < depth_m < 10.0:
                        ray = self.camera_model.projectPixelTo3dRay((center_x, center_y))
                        detection["center_3d"] = {
                            "x": ray[0] * depth_m,
                            "y": ray[1] * depth_m,
                            "z": ray[2] * depth_m,
                        }
             
            detections_list.append(detection)

        self.assign_position_labels(detections_list)

        if self.enable_display:
            self.draw_detections(display_color_img, cv_depth_image, depth_msg.encoding, detections_list)

        json_string = json.dumps(detections_list)
        msg = String()
        msg.data = json_string
        self.detection_pub.publish(msg)
        now = time.time()
        if now - self.last_log_time >= self.log_interval_sec:
            self.get_logger().info(f"已发布 {len(detections_list)} 个检测结果 (JSON)")
            self.last_log_time = now

    def depth_to_meters(self, depth_value, encoding):
        if encoding == '32FC1':
            return depth_value
        return depth_value / 1000.0

    def assign_position_labels(self, detections_list):
        detections_by_class = {}
        for detection in detections_list:
            object_type = detection.get("object_type") or normalize_object_keyword(detection["class_name"])
            if object_type in (BOTTLE, FRUIT):
                detections_by_class.setdefault(object_type, []).append(detection)

        for object_type, detections in detections_by_class.items():
            if len(detections) == 1:
                continue

            detections.sort(key=lambda item: item["center_pixel"][0])
            count = len(detections)

            for index, detection in enumerate(detections):
                if index == 0:
                    label = "left"
                elif index == count - 1:
                    label = "right"
                else:
                    label = "middle"

                detection["position_label"] = label
                detection["position_rank"] = index + 1
                if label == "middle" and count > 3:
                    detection["instance_name"] = f"{object_type}_{label}_{index}"
                else:
                    detection["instance_name"] = f"{object_type}_{label}"

    def draw_detections(self, display_color_img, cv_depth_image, depth_encoding, detections_list):
        for detection in detections_list:
            x1, y1, x2, y2 = detection["box"]
            center_x, center_y = detection["center_pixel"]
            label_text = f"{detection['instance_name']} {detection['confidence']:.2f}"
            if detection["center_3d"] is not None:
                label_text += f" | Z:{detection['center_3d']['z']:.2f}m"

            cv2.rectangle(display_color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display_color_img, (center_x, center_y), 5, (0, 0, 255), -1)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y1 = max(0, y1 - text_h - 10)
            text_y2 = max(text_h + 5, y1)
            cv2.rectangle(display_color_img, (x1, text_y1), (x1 + text_w, text_y2), (0, 255, 0), -1)
            cv2.putText(display_color_img, label_text, (x1, text_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if depth_encoding == '32FC1':
            depth_display = np.clip(cv_depth_image, 0.0, 5.0)
            depth_display = (depth_display / 5.0 * 255).astype(np.uint8)
        else:
            depth_display = np.clip(cv_depth_image, 0, 5000)
            depth_display = (depth_display / 5000.0 * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        depth_colormap_resized = cv2.resize(depth_colormap, (display_color_img.shape[1], display_color_img.shape[0]))
        combined_display = np.hstack((display_color_img, depth_colormap_resized))

        cv2.imshow("YOLOv11 3D Object Detection | Color & Depth", combined_display)
        cv2.waitKey(1)

    # --- 新增：节点销毁时关闭 OpenCV 窗口 ---
    def destroy_node(self):
        if self.enable_display:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = UltimateYoloNodeCPU()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().fatal(f"节点启动或运行中发生致命错误: {e}")
        else:
            print(f"节点初始化失败: {e}")
    finally:
        if node and rclpy.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
