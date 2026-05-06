# interface

这是一个 ROS 2 Python 包，用于相机检测、信号控制和机械臂抓取/释放。

## 构建

```bash
colcon build --packages-select interface
source install/setup.bash
```

## 入口

- `interface.ultimate_yolo_node_cpu`：YOLO + 深度检测节点
- `interface.master_node`：中控节点，处理 1-10 信号
- `interface.action1_4_move`：机械臂平移节点
- `interface.action5_rot`：旋转演示节点
- `interface.action6_gsp`：抓取节点
- `interface.action7_flat`：释放节点
- `interface.action8_rst`：复位节点

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
