import cv2
import numpy as np
import pyorbbecsdk as obs
import time

class CameraStreamViewer:
    """
    一个简单的类，用于初始化奥比托Gemini 330相机，
    并实时显示其彩色和深度视频流。
    
    该版本已根据用户提供的健壮代码进行修正，正确处理了D2C对齐后的深度数据。
    """
    def __init__(self):
        """初始化相机查看器。"""
        self.pipeline = None
        self.config = None
        self._init_camera()

    def _init_camera(self):
        """私有方法，用于初始化和启动奥比托相机。"""
        print("正在初始化奥比托Gemini 330相机...")
        try:
            self.pipeline = obs.Pipeline()
            self.config = obs.Config()
            
            # 1. 配置彩色流
            color_profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
            color_profile = color_profile_list.get_video_stream_profile(640, 480, obs.OBFormat.RGB, 30)
            self.config.enable_stream(color_profile)
            
            # 2. 配置深度流
            depth_profile_list = self.pipeline.get_stream_profile_list(obs.OBSensorType.DEPTH_SENSOR)
            depth_profile = depth_profile_list.get_video_stream_profile(640, 400, obs.OBFormat.Y16, 30)
            self.config.enable_stream(depth_profile)

            # 3. 启用D2C硬件对齐
            self.config.set_align_mode(obs.OBAlignMode.HW_MODE)

            # 4. 启动管道
            self.pipeline.start(self.config)
            print("相机初始化成功！实时视频流即将显示。")
            print("按 'q' 键关闭窗口并退出程序。")
            
        except Exception as e:
            print(f"相机初始化失败，请检查连接或驱动: {e}")
            raise

    def run(self):
        """主运行循环，用于捕获和显示帧。"""
        cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Depth Stream (Colormap)", cv2.WINDOW_AUTOSIZE)
        
        while True:
            frames = self.pipeline.wait_for_frames(100)
            if not frames:
                continue

            # --- 处理彩色图像 ---
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            
            color_image_rgb = np.asanyarray(color_frame.get_data()).reshape((480, 640, 3))
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

            # --- 处理深度图像 (已修正) ---
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # !! 关键修正步骤，借鉴自您的代码 !!
            # 1. 首先，将获取的8位字节数组转换为Numpy数组
            depth_data_uint8 = np.asanyarray(depth_frame.get_data())
            # 2. 然后，用.view()改变对内存的看法，将其视为16位无符号整数数组
            depth_data_uint16 = depth_data_uint8.view(dtype=np.uint16)
            # 3. 现在，可以安全地将这个拥有正确元素数量的数组reshape成二维图像了
            #    由于D2C对齐，深度图尺寸会匹配彩色图 (480, 640)
            depth_image = depth_data_uint16.reshape((480, 640))
            
            # --- 可视化深度图 ---
            # 为了更好的视觉对比度，将深度值限制在一个有效范围内
            min_depth_mm, max_depth_mm = 500, 5000  # 0.5m to 5m
            clipped_depth = np.clip(depth_image, min_depth_mm, max_depth_mm)
            
            # 归一化到0-255以便应用色彩映射
            normalized_depth = cv2.normalize(clipped_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

            # --- 显示图像 ---
            cv2.imshow("Color Stream", color_image_bgr)
            cv2.imshow("Depth Stream (Colormap)", depth_colormap)

            # --- 等待按键退出 ---
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                print("检测到 'q' 键，正在关闭程序...")
                break

        # --- 清理资源 ---
        print("正在释放相机资源...")
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已成功退出。")

def main():
    """程序主入口"""
    try:
        viewer = CameraStreamViewer()
        viewer.run()
    except Exception as e:
        print(f"程序运行时发生错误: {e}")

if __name__ == "__main__":
    main()