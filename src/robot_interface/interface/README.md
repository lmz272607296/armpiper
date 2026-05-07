# interface

这是一个 ROS 2 Python 包，用于相机检测、信号控制和机械臂抓取/释放。

## 构建

```bash
colcon build --packages-select interface
source install/setup.bash
```

## 入口

- `interface.ultimate_yolo_node_cpu`：YOLO + 深度检测节点
- `interface.eye_in_hand_calibration_node`：眼在手上坐标转换节点，将 YOLO 相机坐标转换为 Piper 基座坐标
- `interface.master_node`：中控节点，处理 1-10 信号
- `interface.action1_4_move`：机械臂平移节点
- `interface.action5_rot`：旋转演示节点
- `interface.action6_gsp`：抓取节点
- `interface.action7_flat`：释放节点
- `interface.action8_rst`：复位节点
- `interface.chessboard_hand_eye_calibration_node`：使用棋盘格和 Piper 末端位姿求眼在手上外参
- `interface.camera_base_transform_node`：使用固定相机外参，将 YOLO 相机坐标转换为 Piper 基座坐标
- `interface.piper_keyboard_joint_jog`：标定时使用的 Piper 关节点动键盘节点

## 信号

- `1` 前
- `2` 后
- `3` 左
- `4` 右
- `5` 旋转演示
- `6` 抓取
- `7` 释放
- `8` bottle
- `9` fruit
- `10` 复位

## 物体类型

- `bottle` 对应瓶子
- `fruit` 对应水果
- `cube`、`apple`、`orange`、`furit` 都会归一成 `fruit`

## 话题

- `/recognized_signal`：编号信号输入
- `/grasp_object_type`：当前物体类型
- `/grasp_target_pose`：抓取目标位姿
- `/arm_action_command`：机械臂动作
- `/task_status`：状态输出

## 眼在手上坐标转换

相机安装在 Piper 末端法兰上时，`ultimate_yolo_node_cpu` 输出的 `center_3d` 是相机坐标系坐标。启动转换节点后，会订阅 `/yolo_3d_detections_json` 和 `/end_pose_stamped`，发布 `/yolo_3d_detections_base_json`，其中 `center_3d` 已替换为相对于机械臂 `base_link` 原点的米制坐标。

```bash
ros2 run interface eye_in_hand_calibration_node
```

让中控直接使用基座坐标检测结果：

```bash
ros2 run interface master_node --ros-args -p detection_topic:=/yolo_3d_detections_base_json
```

外参优先从当前工作目录的 `hand_eye_calibration.txt` 读取，文件格式为 4x4 齐次矩阵，含义是 `link6 -> camera_color_optical_frame`。如果没有该文件，会使用参数默认值：

```bash
ros2 run interface eye_in_hand_calibration_node --ros-args \
  -p camera_xyz_in_flange:="[0.07720, -0.0165, 0.09]" \
  -p camera_rpy_in_flange:="[-1.5707963268, 0.0, -1.5707963268]"
```

常用话题：

- 输入相机检测：`/yolo_3d_detections_json`
- 输入末端位姿：`/end_pose_stamped`
- 输出基座检测：`/yolo_3d_detections_base_json`

## 棋盘格手眼标定

相机安装在 Piper 末端时，使用 `chessboard_hand_eye_calibration_node` 采集多组 `base_link -> link6` 末端位姿和 `chessboard -> camera_color_optical_frame` 棋盘格位姿，通过 OpenCV `calibrateHandEye` 求 `link6 -> camera_color_optical_frame`。该流程不需要探针对准棋盘格角点。

需要工具：

- 彩色图：`/camera/camera/color/image_raw`
- 相机内参：`/camera/camera/color/camera_info`
- Piper 末端位姿：`/end_pose_stamped` 或 `/end_pose`
- 棋盘格标定板：默认按 `7 行 11 列完整方块，行方向底部额外有半行` 解释，也就是内部角点 `10 x 8`、方格边长 `20 mm`
- OpenCV 参数填的是内部角点数，不是完整方块数；普通完整棋盘格通常是 `列方块数 - 1` 和 `行方块数 - 1`
- 如果底部半行确实形成了可检测的一整排内部角点，行方向可按 `8` 个内部角点；如果相机窗口无法稳定识别，请改成实际可见的内部角点网格数量

启动标定节点：

```bash
ros2 run interface chessboard_hand_eye_calibration_node --ros-args \
  -p chessboard_inner_corners_x:=10 \
  -p chessboard_inner_corners_y:=8 \
  -p square_size_m:=0.02 \
  -p output_matrix_file:=hand_eye_calibration.txt
```

默认 `display:=false`，节点不会直接弹 OpenCV 窗口，而是发布调试图话题 `/hand_eye_calibration_debug_image`。这是为了避免 Wayland/GNOME 环境下 `cv2.imshow` 导致窗口卡死。

默认也关闭 `findChessboardCornersSB`，因为它在部分环境下会明显卡顿。当前默认使用较轻的 `findChessboardCorners` 路径；如需切换可设置 `-p use_find_chessboard_sb:=true`。

可用 `rqt_image_view` 查看：

```bash
ros2 run rqt_image_view rqt_image_view
```

然后选择话题：

```text
/hand_eye_calibration_debug_image
```

如果你确认当前桌面环境对 OpenCV 弹窗兼容，也可以显式打开：

```bash
ros2 run interface chessboard_hand_eye_calibration_node --ros-args \
  -p chessboard_inner_corners_x:=10 \
  -p chessboard_inner_corners_y:=8 \
  -p square_size_m:=0.02 \
  -p display:=true \
  -p output_matrix_file:=hand_eye_calibration.txt
```

操作方法：

1. 把棋盘格固定在桌面或工作台上，采集期间不能移动。
2. 启动相机和 Piper 驱动：`ros2 launch piper start_single_piper.launch.py`。
3. 确认 `/end_pose_stamped` 或 `/end_pose` 正在更新，确认窗口能完整识别棋盘格。
4. 每移动到一个新的机械臂姿态后，保持棋盘格完整可见，按窗口中的 `c` 采样。
5. 建议采集 12-20 组姿态，姿态要有明显旋转变化，不要只平移或只微小晃动。
6. 按 `s` 求解，结果保存为 `hand_eye_calibration.txt`，可直接供 `eye_in_hand_calibration_node` 使用。

也可以用话题命令采样和求解：

```bash
ros2 topic pub --once /hand_eye_calibration_command std_msgs/msg/String "{data: 'capture'}"
ros2 topic pub --once /hand_eye_calibration_command std_msgs/msg/String "{data: 'solve'}"
```

如果没有示教器，可使用轻量关节点动节点改变姿态：

```bash
ros2 run interface piper_keyboard_joint_jog
```

键位：`A/D=joint1`，`W/S=joint2`，`R/F=joint3`，`T/G=joint5`，`Y/H=joint4`，`U/J=joint6`，`step 2` 可把步长改为 2 度。

运行时把 YOLO 输出转换到基座坐标：

```bash
ros2 run interface eye_in_hand_calibration_node --ros-args \
  -p hand_eye_matrix_file:=hand_eye_calibration.txt
```

然后中控继续使用 `/yolo_3d_detections_base_json`：

```bash
ros2 run interface master_node --ros-args -p detection_topic:=/yolo_3d_detections_base_json
```

## 固定相机到基座转换

如果相机固定在桌面或支架上、不跟随机械臂末端运动，运行时可使用 `camera_base_transform_node` 加载一个已经求好的 `camera_color_optical_frame -> base_link` 矩阵。

运行时把 YOLO 输出转换到基座坐标：

```bash
ros2 run interface camera_base_transform_node --ros-args \
  -p matrix_file:=camera_base_calibration.txt
```
