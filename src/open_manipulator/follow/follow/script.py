import matplotlib.pyplot as plt
import numpy as np
import time

# 假设你的列表有16个元素
data = np.random.rand(21)  # 随机生成16个元素

# 创建图形
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
line, = ax.plot(data, 'r-', marker='o')

# 设置标题和标签
ax.set_title("16 Element Curve")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.set_ylim(0, 1)  # 设置y轴范围，可以根据你的数据调整

# 模拟刷新数据的循环
for _ in range(100):  # 假设刷新100次
    data = np.random.rand(16)  # 重新生成16个数据
    line.set_ydata(data)  # 更新曲线的y值
    plt.draw()  # 刷新图形
    plt.pause(0.1)  # 暂停0.1秒，模拟数据更新的过程

plt.ioff()  # 关闭交互模式
plt.show()  # 显示最终的图形
