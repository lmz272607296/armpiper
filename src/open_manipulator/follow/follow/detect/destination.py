import cv2
import numpy as np
import pyorbbecsdk as obs
import os
import time
from scipy.optimize import least_squares

class CameraHandEyeCalibrator:
    """
    一个用于奥比托（Orbbec）深度相机进行内参和手眼标定的类（最终修正版）。

    该类处理以下任务:
    阶段 1: 内参标定
    1. 初始化和配置Orbbec Gemini 330相机。
    2. 捕获对齐的彩色图像帧，检测棋盘格角点。
    3. 执行相机内参标定计算，并保存结果。

    阶段 2: 手眼标定 (基于单轴旋转的几何方法)
    4. 加载已保存的内参。
    5. 在机械臂旋转不同角度时，通过深度相机精确测量棋盘格原点的3D坐标 (tvec)。
    6. 要求用户输入与每一帧对应的电机角度。
    7. 使用采集到的3D坐标点，通过SVD拟合旋转平面，并通过最小二乘法拟合圆心。
    8. 根据正确的几何约束计算出机械臂基座相对于相机的位姿。
    9. (新增) 提供一个“最佳拟合”结果和一个“Z轴约束”的修正结果。
    10. 将结果转换为用户定义的机器人坐标系并保存。
    """
    def __init__(self, chessboard_x, chessboard_y, square_size_m):
        self.CHESSBOARD_SIZE = (chessboard_x, chessboard_y)
        self.SQUARE_SIZE = square_size_m
        self.INTRINSICS_FILE_NPZ = "orbbec_gemini330_intrinsics.npz"
        self.INTRINSICS_FILE_TXT = "intrinsics_report.txt"
        print(f"棋盘格配置: {self.CHESSBOARD_SIZE} 个内部角点, 方格边长: {self.SQUARE_SIZE * 1000:.2f} mm")

        self.pipeline = None
        self.config = None
        self._init_orbbec_camera()

        self.object_point = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        self.object_point[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.object_point *= self.SQUARE_SIZE
        
        self.obj_points = []
        self.img_points = []
        self.pose_data = []

        self.camera_matrix = None
        self.dist_coeffs = None
        self.current_phase = "intrinsic"

        self._handle_intrinsics_on_startup()

    def _init_orbbec_camera(self):
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
            print("相机初始化成功！请等待5-10分钟让相机预热以获得更稳定的深度数据。")
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise

    def _handle_intrinsics_on_startup(self):
        if os.path.exists(self.INTRINSICS_FILE_NPZ):
            with np.load(self.INTRINSICS_FILE_NPZ) as data:
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
            print("-" * 30)
            print(f"成功加载已有的相机内参: '{self.INTRINSICS_FILE_NPZ}'")
            
            while True:
                choice = input(">>> 您想使用这些已有的内参吗? (y/n): ").lower().strip()
                if choice in ['y', 'n']: break
                print("无效输入，请输入 'y' 或 'n'.")

            if choice == 'y': self.current_phase = "hand-eye"
            else: self.current_phase = "intrinsic"
            print("-" * 30)
        else:
            print("未找到内参文件。请先进行内参标定。")
            self.current_phase = "intrinsic"

    def capture_and_process_frame(self, capture_flag=False):
        frames = self.pipeline.wait_for_frames(100)
        if not frames: return None
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame: return None

        height, width = color_frame.get_height(), color_frame.get_width()
        color_data = np.asanyarray(color_frame.get_data())
        color_image_bgr = cv2.cvtColor(color_data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD_SIZE, None)

        if ret:
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            if capture_flag:
                if self.current_phase == "intrinsic":
                    print(f"找到角点！正在保存第 {len(self.obj_points) + 1} 组[内参]数据...")
                    self.obj_points.append(self.object_point)
                    self.img_points.append(corners_subpix)
                elif self.current_phase == "hand-eye":
                    self._capture_hand_eye_pose(corners_subpix, depth_frame)
            cv2.drawChessboardCorners(color_image_bgr, self.CHESSBOARD_SIZE, corners_subpix, ret)
        
        self._update_display_text(color_image_bgr)
        return color_image_bgr

    def _update_display_text(self, image):
        phase_text = f"Phase: {'INTRINSIC' if self.current_phase == 'intrinsic' else 'HAND-EYE'}"
        count_text = f"Captures: {len(self.obj_points) if self.current_phase == 'intrinsic' else len(self.pose_data)}"
        
        if self.current_phase == 'intrinsic':
            help1 = "Press 'c' to capture. (Need >15 poses)"
            help2 = "Press 's' to save/calibrate."
        else:
            help1 = "Press 'h' to capture. (Need >10 poses, >90 deg range)"
            help2 = "Press 'e' to execute."

        cv2.putText(image, phase_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(image, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, help1, (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(image, help2, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(image, "Press 'q' to quit", (500, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    def _capture_hand_eye_pose(self, corners, depth_frame):
        print("\n捕获手眼标定位姿...")
        try:
            height, width = depth_frame.get_height(), depth_frame.get_width()
            depth_scale = depth_frame.get_depth_scale()
            raw_data = np.asanyarray(depth_frame.get_data())
            depth_image = raw_data.view(np.uint16).reshape((height, width))
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

            origin_corner_uv = corners[0][0]
            u, v = int(round(origin_corner_uv[0])), int(round(origin_corner_uv[1]))

            win_size = 5
            half_win = win_size // 2
            u_min, u_max = max(0, u - half_win), min(width - 1, u + half_win)
            v_min, v_max = max(0, v - half_win), min(height - 1, v + half_win)
            
            depth_patch = depth_image[v_min:v_max+1, u_min:u_max+1]
            valid_depths = depth_patch[depth_patch > 0]
            
            if valid_depths.size < (win_size * win_size / 2):
                print(f"错误: 角点 ({u}, {v}) 周围有效深度点不足。")
                return
            
            stable_raw_depth = np.median(valid_depths)
            depth_in_m = (stable_raw_depth * depth_scale) / 1000.0
            if depth_in_m <= 0: return

            x_m = (u - cx) * depth_in_m / fx
            y_m = (v - cy) * depth_in_m / fy
            tvec_accurate = np.array([x_m, y_m, depth_in_m]).reshape(3, 1)
            print(f"通过深度相机测得的原点坐标 (tvec): {tvec_accurate.flatten().round(4)}")

            angle_str = input(f">>> 请输入当前电机的角度 (例如: 180), 然后按Enter: ")
            angle = float(angle_str)
            
            _, rvec, _ = cv2.solvePnP(self.object_point, corners, self.camera_matrix, self.dist_coeffs)
            self.pose_data.append({'rvec': rvec, 'tvec': tvec_accurate, 'angle': angle})
            print(f"成功保存第 {len(self.pose_data)} 组数据 (角度: {angle}°)。请转动电机。")

        except ValueError: print("输入无效！该点未被保存。")
        except Exception as e: print(f"计算坐标时发生错误: {e}")

    def calibrate_intrinsics(self):
        if len(self.obj_points) < 15:
            print(f"错误: 采集图像数量不足 ({len(self.obj_points)}/15)，请采集更多不同角度的图像。")
            return

        print("正在进行内参标定计算...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, (640, 480), None, None)
        if not ret: 
            print("内参标定失败！")
            return

        self.camera_matrix, self.dist_coeffs = mtx, dist
        mean_error = np.mean([cv2.norm(self.img_points[i], cv2.projectPoints(self.obj_points[i], rvecs[i], tvecs[i], mtx, dist)[0], cv2.NORM_L2)/len(self.img_points[i]) for i in range(len(self.obj_points))])
        print(f"内参标定完成，平均重投影误差: {mean_error:.4f} 像素")

        np.savez(self.INTRINSICS_FILE_NPZ, camera_matrix=mtx, dist_coeffs=dist)
        self._save_intrinsics_to_txt()
        print("\n" + "="*50)
        print("阶段 1 完成: 内参标定成功！")
        self.current_phase = "hand-eye"

    def _save_intrinsics_to_txt(self):
        with open(self.INTRINSICS_FILE_TXT, 'w') as f:
            f.write(f"Camera Intrinsic Calibration Report - {time.asctime()}\n")
            f.write("="*30 + "\n\nCamera Matrix (fx, fy, cx, cy):\n")
            np.savetxt(f, self.camera_matrix, fmt='%.4f')
            f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
            np.savetxt(f, self.dist_coeffs, fmt='%.6f')
        print(f"已生成内参报告: {self.INTRINSICS_FILE_TXT}")

    def calculate_hand_eye_circle_fit(self):
        if len(self.pose_data) < 10:
            print(f"错误: 手眼标定采集的数据点不足 ({len(self.pose_data)}/10)。")
            return
        
        print("\n正在执行手眼标定（基于圆拟合）...")
        points = np.array([p['tvec'].flatten() for p in self.pose_data])

        # 步骤 1: SVD拟合平面，找到旋转轴方向 (平面法向量)，这是“原始Z轴”
        points_centroid = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - points_centroid)
        z_axis_base_raw = vh[2, :]
        print(f"步骤1: 计算出的原始旋转轴方向 (n):\n{np.round(z_axis_base_raw, 4)}")

        # 步骤 2: 拟合圆心
        def error_func(params, points):
            return np.linalg.norm(points - params[:3], axis=1) - params[3]
        initial_params = np.append(np.mean(points, axis=0), np.std(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))
        res = least_squares(error_func, initial_params, args=(points,), method='lm')
        circle_center_calculated = res.x[:3]
        
        # 步骤 3: 【已修正】将圆心投影到它所在的旋转平面上，得到平移向量 t
        t_camera_base = (circle_center_calculated - np.dot(circle_center_calculated - points_centroid, z_axis_base_raw) * z_axis_base_raw).reshape(3, 1)
        print(f"\n步骤2: 修正后的圆心 (t_camera_base):\n{np.round(t_camera_base.flatten(), 4)}")
        
        # 步骤 4: 定义基座的“原始”X, Y, Z轴
        # 我们仍然使用几何方法来构建一个初始的、自洽的右手坐标系
        point_180_data = next((p for p in self.pose_data if p['angle'] == 180), self.pose_data[0])
        if point_180_data['angle'] != 180: print("\n警告: 未找到180度数据点，使用第一个点作为X轴参考。")
        
        x_axis_base_raw = point_180_data['tvec'].flatten() - t_camera_base.flatten()
        x_axis_base_raw = x_axis_base_raw - np.dot(x_axis_base_raw, z_axis_base_raw) * z_axis_base_raw
        x_axis_base_raw /= np.linalg.norm(x_axis_base_raw)
        
        y_axis_base_raw = np.cross(z_axis_base_raw, x_axis_base_raw)

        # 步骤 5: 构建从基座到相机的“原始”旋转矩阵
        R_base_camera_raw = np.stack([x_axis_base_raw, y_axis_base_raw, z_axis_base_raw], axis=1) # 按列堆叠

        # ======================= 核心修正步骤 =======================
        # 根据您的观察，计算出的坐标系需要绕其自身的X轴旋转180度，才能与您期望的坐标系对齐。
        # (Y -> -Y, Z -> -Z)
        # 绕X轴旋转180度的修正矩阵
        print("\n步骤3: 应用坐标系姿态修正...")
        R_correction = np.array([[1,  0,  0],
                                 [0, -1,  0],
                                 [0,  0, -1]])

        # 将原始旋转矩阵乘以修正矩阵，得到最终的、符合您期望的旋转矩阵
        R_base_camera_final = R_base_camera_raw @ R_correction
        print("姿态修正完成，Y轴和Z轴已被翻转。")
        # ==========================================================

        # 步骤 6: 构建最终的、完整的变换矩阵
        # 这个矩阵 T_base_<-_cam 用于将点从相机坐标系变换到基座坐标系
        T_base_from_cam = np.identity(4)
        T_base_from_cam[:3, :3] = R_base_camera_final.T
        T_base_from_cam[:3, 3] = (-R_base_camera_final.T @ t_camera_base).flatten()


        print("\n" + "="*50)
        print("【结果 A: 最佳拟合解 (Best Fit Solution)】")
        print("这是最符合您采集数据的数学最优解。")
        
        self.process_and_save_results(T_base_from_cam, "hand_eye_calibration_result_best_fit.npz")
        
        print("\n" + "="*50)
        print("【结果 B: Z轴约束解 (Z-Constrained Solution)】")
        print("此解强制相机与基座在用户坐标系下的Z值(高度)为0。")
        
        self.process_and_save_results(T_base_from_cam, "hand_eye_calibration_result_z_constrained.npz", constrain_z=True)
        print("="*50)

    def process_and_save_results(self, base_H_camera, filename, constrain_z=False):
        """
        【已修正】辅助函数，用于处理坐标系转换、打印和保存结果。
        该函数现在直接使用计算出的 T_base_<-_cam 矩阵，并根据用户坐标系定义来解释和打印它。
        它不再执行额外的矩阵变换。
        """
        # base_H_camera 就是我们需要的最终矩阵 T_base_<-_cam
        # 它将一个点从相机坐标系转换到机械臂基座坐标系。
        final_transform_matrix = base_H_camera.copy()

        # --------------------------------------------------------------------------
        # 核心修正：不再执行 base_H_camera @ T_C_to_U_coord_transform
        # 我们只是为了打印和理解，才定义坐标系。
        # --------------------------------------------------------------------------

        # 对Z轴的约束是施加在最终的平移向量上的。
        # 在我们的定义中，机械臂基座的Z轴（向上）对应变换矩阵平移向量的第三个元素。
        if constrain_z:
            # final_transform_matrix 的第3列是平移向量 t
            # t 的第3个元素 (index 2) 对应的是Z轴（向上）的值
            final_transform_matrix[2, 3] = 0.0
            print("Z轴约束已应用：最终矩阵的Z平移强制为0。")


        print("变换矩阵 (相机坐标系 -> 机械臂基座坐标系):")
        print("说明: 该矩阵用于将相机坐标系下的点转换为机械臂基座坐标系下的点。")
        print("机械臂基座坐标系定义: +X=向前(深度), +Y=向左, +Z=向上")
        print(np.round(final_transform_matrix, 4))

        # 这里保存的 base_H_robot_coord 就是我们最终要用的矩阵
        # 另一个 base_H_camera_coord 实际上是多余的，但为了兼容性保留
        np.savez(filename, 
                 base_H_robot_coord=final_transform_matrix, 
                 base_H_camera_coord=base_H_camera) # base_H_camera 是未应用Z约束的原始矩阵
        print(f"结果已保存至: {os.path.abspath(filename)}")

    def run(self):
        cv2.namedWindow("Camera Hand-Eye Calibration", cv2.WINDOW_AUTOSIZE)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            
            capture_action = False
            if self.current_phase == 'intrinsic':
                if key == ord('c'): capture_action = True
                elif key == ord('s'): self.calibrate_intrinsics()
            else:
                if key == ord('h'): capture_action = True
                elif key == ord('e'): self.calculate_hand_eye_circle_fit()

            display_image = self.capture_and_process_frame(capture_action)
            if display_image is not None:
                cv2.imshow("Camera Hand-Eye Calibration", display_image)
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main():
    # --- 用户配置 ---
    # 建议：如果视角有限，请使用更小的标定板以获得更大的旋转角度。
    # 旋转角度 > 90度 是最重要的！
    CHESSBOARD_X_CORNERS = 7
    CHESSBOARD_Y_CORNERS = 8
    SQUARE_SIZE_IN_METERS = 0.02 # 缩小标定板尺寸以增加旋转范围

    try:
        calibrator = CameraHandEyeCalibrator(CHESSBOARD_X_CORNERS, CHESSBOARD_Y_CORNERS, SQUARE_SIZE_IN_METERS)
        calibrator.run()
    except Exception as e:
        print(f"程序启动时发生严重错误: {e}")

if __name__ == "__main__":
    main()