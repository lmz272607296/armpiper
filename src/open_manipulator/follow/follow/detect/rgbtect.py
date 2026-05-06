import cv2
import numpy as np
import pyorbbecsdk as obs

# CubeDetector 类无需任何修改，它的设计很棒，可以接收任意尺寸的图像。
class CubeDetector:
    """
    一个使用OpenCV进行正方体检测的类。
    它接收一个彩色图像，并尝试找到其中最像正方体轮廓的四边形。
    """
    def __init__(self):
        """
        初始化检测器。可以在这里设置可调参数。
        """
        self.min_area = 2000
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.approx_poly_factor = 0.04

    def detect(self, color_image):
        """
        在输入的彩色图像中检测正方体。
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        processed_image = color_image.copy()
        found_cube_info = None

        if contours:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in sorted_contours:
                if cv2.contourArea(cnt) < self.min_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, self.approx_poly_factor * perimeter, True)

                if len(approx) == 4:
                    vertices = approx.reshape(4, 2)
                    rect = cv2.minAreaRect(cnt)
                    (x, y), (width, height), angle = rect
                    side_length_px = (width + height) / 2
                    found_cube_info = {
                        "vertices": vertices,
                        "side_length_px": side_length_px,
                        "center_2d": (int(x), int(y))
                    }
                    cv2.drawContours(processed_image, [approx], -1, (0, 255, 0), 3)
                    for vertex in vertices:
                        cv2.circle(processed_image, tuple(vertex), 5, (255, 0, 0), -1)
                    cv2.circle(processed_image, found_cube_info["center_2d"], 5, (0, 0, 255), -1)
                    break
        return processed_image, found_cube_info


class CameraStreamViewer:
    """
    集成了带ROI选择功能的正方体检测的相机流查看器。
    """
    def __init__(self):
        """初始化相机、检测器以及ROI相关的状态变量。"""
        self.pipeline = None
        self.config = None
        self.detector = CubeDetector()
        
        # --- 新增：ROI状态变量 ---
        self.roi_start_point = None
        self.roi_end_point = None
        self.is_drawing = False
        self.roi_selected = False
        
        self._init_camera()

    # --- 新增：鼠标回调函数 ---
    def _mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件以绘制ROI。"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 如果按下左键，重置ROI并记录起点
            self.roi_selected = False
            self.is_drawing = True
            self.roi_start_point = (x, y)
            self.roi_end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            # 如果正在绘制，更新矩形的终点
            if self.is_drawing:
                self.roi_end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # 如果松开左键，完成绘制
            if self.is_drawing:
                self.is_drawing = False
                self.roi_end_point = (x, y)
                # 只有当矩形有一定大小时，才确认ROI选择成功
                if abs(self.roi_start_point[0] - self.roi_end_point[0]) > 10 and \
                   abs(self.roi_start_point[1] - self.roi_end_point[1]) > 10:
                    self.roi_selected = True
                else: # 如果只是点击了一下，则取消选择
                    self.roi_start_point = None
                    self.roi_end_point = None

    # --- 新增：获取标准化的ROI矩形(x, y, w, h) ---
    def _get_roi_rect(self):
        """根据起点和终点，计算并返回标准化的ROI矩形(x,y,w,h)"""
        if not self.roi_start_point or not self.roi_end_point:
            return None
        
        x1, y1 = self.roi_start_point
        x2, y2 = self.roi_end_point
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        
        return (x, y, w, h)


    def _init_camera(self):
        """私有方法，用于初始化和启动奥比托相机。"""
        print("正在初始化奥比托Gemini 330相机...")
        try:
            self.pipeline = obs.Pipeline()
            self.config = obs.Config()
            
            color_profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_video_stream_profile(640, 480, obs.OBFormat.RGB, 30)
            self.config.enable_stream(color_profile)
            
            depth_profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_video_stream_profile(640, 400, obs.OBFormat.Y16, 30)
            self.config.enable_stream(depth_profile)

            self.config.set_align_mode(obs.OBAlignMode.HW_MODE)
            self.pipeline.start(self.config)
            print("相机初始化成功！")
            # --- 修改：更新了操作提示 ---
            print("=== 操作指南 ===")
            print("- 在彩色图像窗口中'左键拖拽'来选择一个区域进行拟合。")
            print("- 按 'r' 键重置选择区域。")
            print("- 按 'q' 键关闭窗口并退出程序。")
            
        except Exception as e:
            print(f"相机初始化失败，请检查连接或驱动: {e}")
            raise

    def run(self):
        """主运行循环，捕获、处理和显示帧。"""
        window_name = "Color Stream with Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth Stream (Colormap)", cv2.WINDOW_AUTOSIZE)
        
        # --- 新增：绑定鼠标回调函数到窗口 ---
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if not frames: continue

            color_frame = frames.get_color_frame()
            if not color_frame: continue
            
            color_image_rgb = np.asanyarray(color_frame.get_data()).reshape((480, 640, 3))
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

            depth_frame = frames.get_depth_frame()
            if not depth_frame: continue
            
            depth_data_uint8 = np.asanyarray(depth_frame.get_data())
            depth_data_uint16 = depth_data_uint8.view(dtype=np.uint16)
            depth_image = depth_data_uint16.reshape((480, 640))
            
            # 创建一个用于显示的图像副本
            display_image = color_image_bgr.copy()

            # --- 修改：检测逻辑现在基于ROI状态 ---
            if self.roi_selected:
                # 1. 获取标准化的ROI矩形
                roi_rect = self._get_roi_rect()
                if roi_rect:
                    rx, ry, rw, rh = roi_rect
                    
                    # 绘制黄色的、固定的ROI框
                    cv2.rectangle(display_image, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
                    
                    # 2. 从原图中裁剪出ROI
                    # 添加边界检查，防止裁剪区域超出图像范围
                    roi_image = color_image_bgr[ry:ry+rh, rx:rx+rw]
                    
                    # 3. 在ROI内部进行检测 (只在ROI有效时进行)
                    if roi_image.size > 0:
                        processed_roi, cube_info = self.detector.detect(roi_image)
                        
                        # 4. 如果在ROI中检测到物体，转换坐标并显示
                        if cube_info:
                            # --- 坐标转换：将ROI内的局部坐标转换回全局坐标 ---
                            cube_info["center_2d"] = (cube_info["center_2d"][0] + rx, cube_info["center_2d"][1] + ry)
                            cube_info["vertices"] = cube_info["vertices"] + np.array([rx, ry])
                            
                            # 在主显示图像上绘制结果
                            cv2.drawContours(display_image, [cube_info["vertices"]], -1, (0, 255, 0), 3)
                            for vertex in cube_info["vertices"]:
                                cv2.circle(display_image, tuple(vertex), 5, (255, 0, 0), -1)
                            cv2.circle(display_image, cube_info["center_2d"], 5, (0, 0, 255), -1)

                            # 打印信息（使用转换后的全局坐标）
                            center_x, center_y = cube_info["center_2d"]
                            if 0 <= center_y < depth_image.shape[0] and 0 <= center_x < depth_image.shape[1]:
                                depth_mm = depth_image[center_y, center_x]
                                cube_info["depth_mm"] = float(depth_mm)
                                print(
                                    f"检测到正方体! "
                                    f"中心点(2D): {cube_info['center_2d']}, "
                                    f"中心点深度: {cube_info['depth_mm'] / 1000:.3f} m, "
                                    f"估算边长: {cube_info['side_length_px']:.2f} px"
                                )

            # --- 新增：绘制动态的ROI选择框 ---
            if self.is_drawing and self.roi_start_point and self.roi_end_point:
                # 绘制蓝色的、正在拖拽的框
                cv2.rectangle(display_image, self.roi_start_point, self.roi_end_point, (255, 100, 0), 2)


            # --- 可视化深度图 ---
            min_depth_mm, max_depth_mm = 500, 5000
            clipped_depth = np.clip(depth_image, min_depth_mm, max_depth_mm)
            normalized_depth = cv2.normalize(clipped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

            # --- 显示图像 ---
            cv2.imshow(window_name, display_image)
            cv2.imshow("Depth Stream (Colormap)", depth_colormap)
            
            # --- 修改：增加 'r' 键用于重置 ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("检测到 'q' 键，正在关闭程序...")
                break
            elif key == ord('r'):
                print("检测到 'r' 键，已重置ROI选择。请重新拖拽鼠标选择区域。")
                self.roi_selected = False
                self.is_drawing = False
                self.roi_start_point = None
                self.roi_end_point = None

        print("正在释放相机资源...")
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已成功退出。")

def main():
    try:
        viewer = CameraStreamViewer()
        viewer.run()
    except Exception as e:
        print(f"程序运行时发生错误: {e}")

if __name__ == "__main__":
    main()