# ===============================================================================================
# ======================== 开始：一欧元滤波器“大杀器”模块 ========================
# ===============================================================================================
import time
import math
# 注意：numpy 已经在你的代码中导入，这里只是为了模块的完整性
import numpy as np 

# 低通滤波器辅助类
class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.s = None

    def __call__(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        if self.y is None:
            self.s = x
        else:
            self.s = self.alpha * x + (1.0 - self.alpha) * self.y
        self.y = self.s
        return self.s

# 一欧元滤波器核心实现
class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be > 0")
        if min_cutoff <= 0:
            raise ValueError("min_cutoff should be > 0")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff should be > 0")

        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(d_cutoff))
        self.last_time = None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if self.last_time is None and t is not None:
            self.last_time = t
        
        if t is None:
            t = time.time()
            
        dt = t - self.last_time if self.last_time is not None else 1.0
        
        if dt <= 1e-5: # Avoid division by zero
            return x

        self.freq = 1.0 / dt
        self.last_time = t
        
        # Estimate the derivative
        dx = (x - self.x_filter.y) / dt if self.x_filter.y is not None else 0.0
        
        # Filter the derivative
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff))
        
        # Use the filtered derivative to update the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        
        # Filter the signal
        return self.x_filter(x, self._alpha(cutoff))

# 应用于MediaPipe所有关节点的“平滑器”封装类
class LandmarkSmoother:
    def __init__(self, num_landmarks, num_dims=3, freq=30.0, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        """
        为所有关节点初始化平滑器。
        num_landmarks: 要平滑的关节点数量 (例如, 手=21, 姿态=33).
        num_dims: 每个关节点的维度 (2D为2, 3D为3).
        freq: 输入信号的频率 (例如, 相机FPS).
        min_cutoff, beta, d_cutoff: 一欧元滤波器的参数.
        """
        self.num_landmarks = num_landmarks
        self.num_dims = num_dims
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        
        # 为每个关节点的每个维度初始化一个滤波器
        self.filters = [[OneEuroFilter(freq, min_cutoff, beta, d_cutoff) for _ in range(num_dims)] for _ in range(num_landmarks)]
        
        #【新增】用于存储上一次有效的平滑结果
        self.last_smoothed_landmarks = None

    def __call__(self, landmarks_np):
        """
        对numpy数组格式的关节点数据应用平滑。
        landmarks_np: 一个 (num_landmarks, num_dims) 的numpy数组，无效点可以是np.nan
        """
        if landmarks_np is None:
            return self.last_smoothed_landmarks # 如果输入为空，返回上一次的结果

        if self.last_smoothed_landmarks is None:
             # 如果是第一次，用当前值初始化
             self.last_smoothed_landmarks = np.copy(landmarks_np)
        
        smoothed_landmarks = np.zeros_like(landmarks_np)
        current_time = time.time()
        
        for i in range(self.num_landmarks):
            for j in range(self.num_dims):
                value = landmarks_np[i, j]
                
                #【关键修改】处理无效点
                if np.isnan(value):
                    # 如果当前值无效，使用上一次的平滑值
                    smoothed_landmarks[i, j] = self.last_smoothed_landmarks[i, j]
                else:
                    smoothed_landmarks[i, j] = self.filters[i][j](value, current_time)

        # 更新历史记录
        self.last_smoothed_landmarks = np.copy(smoothed_landmarks)
        return smoothed_landmarks

# ===============================================================================================
# ========================= 结束：一欧元滤波器“大杀器”模块 =========================
# ===============================================================================================