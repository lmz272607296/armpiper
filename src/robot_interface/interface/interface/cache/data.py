import numpy as np

# 假设您的 .npy 文件名为 'your_file.npy'
# 请将其替换为您的实际文件名
file_path = 'ping.npy'

# 加载 .npy 文件
try:
    data = np.load(file_path)

    # 打印原始数组的形状
    print(f"原始数组的形状: {data.shape}")

    # 检查原始形状是否符合预期
    if data.shape == (1, 17):
        # 删除第17列 (索引为16)
        # 我们通过切片选取所有行和除了最后一列之外的所有列
        new_data = data[:, :-1]

        # 打印新数组的形状
        print(f"新数组的形状: {new_data.shape}")

        # 可选：将新数组保存为新的 .npy 文件
       # np.save('hello16.npy', new_data)
        # print("已将新数组保存为 'new_file.npy'")

    else:
        print("文件的形状不是 (374, 17)，请检查您的文件。")

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保文件路径正确。")
except Exception as e:
    print(f"处理文件时发生错误: {e}")