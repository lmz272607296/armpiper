# 文件名: 3dpoint_optimized.py
# 描述: 使用最高分辨率(1280x800)并优化了代码结构和算法参数的最终版本。

import cv2
import numpy as np
import pyorbbecsdk as obs
import open3d as o3d
import time
from itertools import combinations

# --- 全局变量用于鼠标框选 (无改动) ---
drawing = False
rect_start_point = None
rect_end_point = None
selection_done = False

def draw_rectangle(event, x, y, flags, param):
    """鼠标回调函数，用于在彩色图像上绘制矩形。"""
    global drawing, rect_start_point, rect_end_point, selection_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rect_start_point = (x, y)
        rect_end_point = None
        selection_done = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end_point = (x, y)
        if rect_start_point and rect_end_point and \
           rect_start_point[0] != rect_end_point[0] and \
           rect_start_point[1] != rect_end_point[1]:
            selection_done = True
            print(f"框选完成: 从 {rect_start_point} 到 {rect_end_point}")

class CubeFitter:
    def __init__(self):
        # --- 新增：分辨率成员变量 ---
        self.color_width = 0
        self.color_height = 0
        self.depth_width = 0
        self.depth_height = 0
        
        self.pipeline = None
        self.config = None
        self.camera_params = None
        self.point_cloud_filter = None
        
        self._init_camera()
        
        cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth Stream (Colormap)", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Color Stream", draw_rectangle)

    def _init_camera(self):
        """初始化奥比托相机，使用最高可用分辨率并优化配置。"""
        print("正在初始化奥比托相机...")
        try:
            self.pipeline = obs.Pipeline()
            self.config = obs.Config()

            # # --- 优化：查询并打印彩色和深度传感器的可用配置 ---
            # self._print_stream_profiles(obs.OBSensorType.COLOR_SENSOR, "彩色")
            # self._print_stream_profiles(obs.OBSensorType.DEPTH_SENSOR, "深度")

            # --- 关键修改：设置高质量分辨率 ---
            # 选择一个支持的高分辨率彩色配置。硬件对齐需要帧率(fps)一致。
            # 这里我们选择 1280x720 @ 30fps。如果您的相机不支持，请根据上面打印的列表修改。
            color_profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_video_stream_profile(1280, 720, obs.OBFormat.RGB, 30)
            if not color_profile:
                raise ValueError("未找到指定的彩色相机配置 (1280x720 @ 30fps)。请检查上方打印的支持列表并修改代码。")
            self.config.enable_stream(color_profile)
            self.color_width = color_profile.get_width()
            self.color_height = color_profile.get_height()

            # 选择最高质量的深度配置：1280x800 @ 30fps
            depth_profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_video_stream_profile(1280, 800, obs.OBFormat.Y16, 30)
            if not depth_profile:
                raise ValueError("未找到指定的深度相机配置 (1280x800 @ 30fps)。请检查上方打印的支持列表。")
            self.config.enable_stream(depth_profile)
            self.depth_width = depth_profile.get_width()
            self.depth_height = depth_profile.get_height()

            print(f"\n配置选择: 彩色({self.color_width}x{self.color_height}), 深度({self.depth_width}x{self.depth_height}) @ 30fps")

            # 关键：硬件对齐模式会将深度图对齐到彩色图
            self.config.set_align_mode(obs.OBAlignMode.HW_MODE)
            self.pipeline.start(self.config)
            
            self.camera_params = self.pipeline.get_camera_param()
            
            print("正在创建SDK点云过滤器...")
            self.point_cloud_filter = obs.PointCloudFilter()
            self.point_cloud_filter.set_camera_param(self.camera_params)
            print("相机和点云过滤器初始化成功！")

        except Exception as e:
            print(f"相机或过滤器初始化失败: {e}")
            raise

    # def _print_stream_profiles(self, sensor_type, sensor_name):
    #     """一个辅助函数，用于打印指定传感器支持的所有流配置。"""
    #     profile_list = self.pipeline.get_stream_profile_list(sensor_type)
    #     print(f"\n--- 可用的 {sensor_name} 分辨率 ---")
    #     for i in range(profile_list.get_count()):
    #         profile = profile_list.get_profile(i)
    #         print(f"  - {profile.get_width()}x{profile.get_height()} @ {profile.get_fps()}fps, Format: {profile.get_format()}")
    #     print("--------------------------")

    def run(self):
        """主运行循环。"""
        global selection_done, rect_start_point, rect_end_point
        print("\n" + "="*50)
        print("操作指南:")
        print("1. 在 'Color Stream' 窗口中，按住鼠标左键并拖动以框选一个目标物体。")
        print("2. 松开鼠标后，程序将开始处理该区域。")
        print("3. 按 'q' 键退出程序。")
        print("="*50 + "\n")

        while True:
            frames = self.pipeline.wait_for_frames(100)
            if not frames: continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame: continue
            
            # --- 优化：使用成员变量进行reshape ---
            # 对齐后，彩色和深度帧的尺寸都与彩色相机配置一致
            h, w = self.color_height, self.color_width
            
            color_image_rgb = np.asanyarray(color_frame.get_data()).reshape((h, w, 3))
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
            
            depth_data_uint16 = np.asanyarray(depth_frame.get_data()).view(np.uint16).reshape((h, w))
            clipped_depth = np.clip(depth_data_uint16, 500, 5000) # mm
            normalized_depth = cv2.normalize(clipped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
            
            # 为了便于观察，可以缩小显示的窗口
            display_bgr = cv2.resize(color_image_bgr, (w // 2, h // 2))
            display_depth = cv2.resize(depth_colormap, (w // 2, h // 2))

            if rect_start_point and rect_end_point:
                # 在缩小的图像上绘制矩形需要转换坐标
                sp = (rect_start_point[0] // 2, rect_start_point[1] // 2)
                ep = (rect_end_point[0] // 2, rect_end_point[1] // 2)
                cv2.rectangle(display_bgr, sp, ep, (0, 255, 0), 2)
            
            cv2.imshow("Color Stream", display_bgr)
            cv2.imshow("Depth Stream (Colormap)", display_depth)
            
            if selection_done:
                final_selection = (rect_start_point, rect_end_point)
                selection_done = False
                
                self.process_selection(frames, final_selection)
                
                rect_start_point, rect_end_point = None, None
                print("\n处理完成。您可以继续框选新的目标，或按 'q' 退出。")

            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
        
        self.cleanup()

    def process_selection(self, frames, selection_box):
        """根据用户的框选区域处理点云。"""
        pcd = self.create_pcd_with_sdk(frames, selection_box)
        if pcd is None or not pcd.has_points():
            print("警告: 框选区域内没有有效的深度点，无法创建点云。")
            return
            
        print(f"从框选区域成功提取 {len(pcd.points)} 个点。")
        
        print("正在进行离群点移除...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_denoised = pcd.select_by_index(ind)
        print(f"移除离群点后剩余 {len(pcd_denoised.points)} 个点。")

        pcd_downsampled = pcd_denoised.voxel_down_sample(voxel_size=0.005)
        print(f"下采样后剩余 {len(pcd_downsampled.points)} 个点。")
        
        self.fit_cube_and_visualize(pcd_downsampled)

    def create_pcd_with_sdk(self, frames, selection_box):
        """使用 pyorbbecsdk 生成点云，并根据选框裁剪。(代码逻辑无改动，但处理的数据分辨率更高了)"""
        print("正在使用SDK生成点云...")
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            print("错误: 无法在帧集合中找到深度帧。")
            return None
        
        # 对齐后的帧尺寸等于彩色帧尺寸
        height = depth_frame.get_height()
        width = depth_frame.get_width()
        
        points_frame = self.point_cloud_filter.process(frames)
        points_data = np.frombuffer(points_frame.get_data(), dtype=np.float32)
        points_xyz = points_data.reshape((height, width, 3))

        x1, y1 = selection_box[0]
        x2, y2 = selection_box[1]
        x_start, x_end = min(x1, x2), max(x1, x2)
        y_start, y_end = min(y1, y2), max(y1, y2)
        
        cropped_points = points_xyz[y_start:y_end, x_start:x_end]
        reshaped_points = cropped_points.reshape(-1, 3)
        valid_points = reshaped_points[np.any(reshaped_points != 0, axis=1)]
        
        if valid_points.shape[0] == 0:
            return None
            
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        return pcd

    def fit_cube_and_visualize(self, pcd):
        """使用RANSAC拟合平面，并计算几何关系。"""
        remaining_pcd = pcd
        detected_planes = []
        
        # 使用高分辨率数据时，可以适当增加对平面内点数量的要求
        min_points_for_plane = 100
        for _ in range(3):
            if len(remaining_pcd.points) < min_points_for_plane: break
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            if len(inliers) < min_points_for_plane: continue
            
            # 保存平面模型和该平面的内点点云
            inlier_pcd = pcd.select_by_index(inliers)
            detected_planes.append((plane_model, inlier_pcd))
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

        if len(detected_planes) < 3:
            print("未能检测到足够的平面（需要3个）来定义一个角点。")
            o3d.visualization.draw_geometries([pcd], window_name="未能拟合 - 原始点云")
            return
            
        print(f"成功检测到 {len(detected_planes)} 个平面。正在分析几何关系...")

        # --- 关键修改：调整垂直度容忍度 ---
        # 0.7的容忍度太高了。0.3 (~17度偏差) 是一个更合理的起点。
        # 0.25 (~14度偏差) 更严格。可以根据实际效果调整。
        tolerance = 0.3
        orthogonal_planes = None
        for combo in combinations(detected_planes, 3):
            normals = [p[0][:3] for p in combo]
            n1, n2, n3 = normals[0], normals[1], normals[2]
            if abs(np.dot(n1, n2)) < tolerance and abs(np.dot(n1, n3)) < tolerance and abs(np.dot(n2, n3)) < tolerance:
                orthogonal_planes = combo
                break
        
        if orthogonal_planes is None:
            print("在检测到的平面中未能找到一组相互近似垂直的平面。")
            o3d.visualization.draw_geometries([pcd], window_name="未能拟合 - 原始点云")
            return
            
        print("找到一组正交平面！正在计算顶点...")
        planes_data = [p[0] for p in orthogonal_planes]
        A = np.array([p[:3] for p in planes_data])
        b = np.array([-p[3] for p in planes_data])
        try:
            corner_vertex = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("计算顶点失败：平面矩阵是奇异的。")
            return

        distances = []
        for i in range(3):
            obb = orthogonal_planes[i][1].get_oriented_bounding_box()
            dims = sorted(obb.extent)
            if len(dims) > 1: distances.append(dims[1])
        
        if not distances:
             print("无法从平面尺寸估算边长。")
             return
        side_length = np.mean(distances)

        print("\n" + "="*20 + " 拟合结果 " + "="*20)
        print(f"推断的边长 (米): {side_length:.4f}")

        dir1, dir2, dir3 = A / np.linalg.norm(A, axis=1, keepdims=True)
        vertices = [
            corner_vertex, corner_vertex + side_length * dir1,
            corner_vertex + side_length * dir2, corner_vertex + side_length * dir3,
            corner_vertex + side_length * (dir1 + dir2), corner_vertex + side_length * (dir1 + dir3),
            corner_vertex + side_length * (dir2 + dir3), corner_vertex + side_length * (dir1 + dir2 + dir3)
        ]
        
        print("推断的8个顶点位置 (x, y, z) 米:")
        for i, v in enumerate(vertices):
            print(f"  顶点 {i}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
        print("="*54)

        self.visualize_results(pcd, vertices, orthogonal_planes)

    def visualize_results(self, original_pcd, vertices, planes):
        """使用Open3D可视化结果。"""
        lines = [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]
        colors = [[1, 0, 0] for _ in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertices),
            lines=o3d.utility.Vector2iVector(lines))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        colored_planes = []
        plane_colors = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]]
        for i, (_, plane_pcd) in enumerate(planes):
            plane_pcd.paint_uniform_color(plane_colors[i])
            colored_planes.append(plane_pcd)

        corner_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        corner_sphere.translate(vertices[0]).paint_uniform_color([0, 1, 1])

        print("\n正在打开可视化窗口... 关闭该窗口后程序将继续等待新的框选。")
        o3d.visualization.draw_geometries(
            [original_pcd, line_set, corner_sphere] + colored_planes,
            window_name="正方体拟合结果", width=800, height=600)

    def cleanup(self):
        """清理资源。"""
        print("正在释放相机资源...")
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已成功退出。")

if __name__ == "__main__":
    try:
        fitter = CubeFitter()
        fitter.run()
    except Exception as e:
        print(f"程序运行时发生致命错误: {e}")