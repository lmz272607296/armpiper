import numpy as np
import os

data_point_4 = [0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# --- 2. 将这四个列表组合成一个列表的列表 ---
trajectory_list = [
    data_point_4
]

# --- 3. 将列表的列表转换为 NumPy 数组 ---
trajectory_data = np.array(trajectory_list)

# --- 4. 打印并确认数组的形状 ---
print(f"创建的轨迹数据形状为: {trajectory_data.shape}")
# 预期的输出应该是: 创建的轨迹数据形状为: (4, 16)
if trajectory_data.shape == (1, 16):
    print("形状正确！")
else:
    print(f"警告！形状不正确，请检查您的输入数据！")

# --- 5. 定义要保存的文件路径 ---
output_dir = os.path.dirname(__file__)
file_name = "ping.npy"
full_path = os.path.join(output_dir, file_name)

# 确保目录存在
os.makedirs(output_dir, exist_ok=True)

# --- 6. 保存 NumPy 数组到 .npy 文件 ---
np.save(full_path, trajectory_data)

print(f"\n形状为 (1, 16) 的轨迹已成功保存到: {full_path}")
