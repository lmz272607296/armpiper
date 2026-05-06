import numpy as np

def base_to_camera(base_coords):
    # 提取旋转矩阵和平移向量
    T_base_to_camera = np.array([
    [0, 0, 9.828e-01, 0.07720],
    [-9.829e-01, 0, 0, -0.0165],
    [0, -9.999e-01, 0, 0],
    [0.00, 0.00, 0.00, 1.00]
  ])
    
    # 备用矩阵

    # T_base_to_camera=np.array([
    #  [-0.1815,  0.0489,  0.9822,  0.0787],
    #  [-0.9833,  0.0012, -0.1818, -0.017 ],
    #  [-0.01  , -0.9988,  0.0479, -0.0012],
    #  [ 0.  ,    0.  ,    0.  ,     1.    ]])

    # T_base_to_camera=np.array([
    #  [-1.841e-01 , 1.390e-02 , 9.828e-01 , 7.720e-02],
    #  [-9.829e-01 , 2.400e-03 , -1.841e-01  ,-1.650e-02],
    #  [ -2.000e-04 , -9.999e-01, -1.410e-02  ,  0],
    #  [ 0.000e+00 , 0.000e+00 , 0.000e+00  ,1.000e+00]])


    rotation_matrix = T_base_to_camera[:3, :3]  # 左上角3x3的旋转矩阵
    translation_vector = T_base_to_camera[:3, 3]  # 最右边一列的平移向量
    
    # 旋转基部坐标
    rotated_coords = np.dot(rotation_matrix, base_coords)
    
    # 平移处理：减去平移向量
    camera_coords = rotated_coords + translation_vector
    
    return camera_coords



base_coords = np.array([0.1, 0.2, 0.3])  # 基部坐标
camera_coords = base_to_camera(base_coords)
print("相机坐标：", camera_coords)
