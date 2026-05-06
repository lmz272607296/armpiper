import pyorbbecsdk as obs
import numpy as np

def get_and_print_factory_intrinsics():
    """
    一个简单的脚本，用于初始化Orbbec相机，
    直接从SDK读取出厂设置的内参，并将其打印出来。
    """
    pipeline = None
    try:
        print("正在连接Orbbec相机以读取出厂内参...")
        
        # 1. 初始化 Pipeline 和 Config
        pipeline = obs.Pipeline()
        config = obs.Config()

        # 2. 获取彩色相机的数据流配置 (Profile)
        # 我们只需要这个profile对象来访问其内参，无需真的启动数据流
        profile_list = pipeline.get_stream_profile_list(obs.OBSensorType.COLOR_SENSOR)
        
        # 选择一个常见的分辨率，内参通常对同一型号相机是固定的
        # 这里我们假设使用640x480，和您标定时一致
        color_profile = profile_list.get_video_stream_profile(640, 480, obs.OBFormat.RGB, 30)
        
        if not color_profile:
            print("错误: 无法找到 640x480 RGB 视频流配置。")
            return

        # 3. 关键步骤：从 Profile 中获取内参对象
        intrinsics = color_profile.get_intrinsic()

        # 4. 将内参格式化为标准的3x3矩阵并打印
        factory_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ])

        print("\n" + "="*40)
        print("成功读取相机出厂内参 (Factory Intrinsics):")
        print("="*40)
        print("相机内参矩阵 (Camera Matrix):")
        print(factory_matrix)
        print(f"\n参数解析:")
        print(f"  fx (焦距 x): {intrinsics.fx:.4f}")
        print(f"  fy (焦距 y): {intrinsics.fy:.4f}")
        print(f"  cx (主点 x): {intrinsics.cx:.4f}")
        print(f"  cy (主点 y): {intrinsics.cy:.4f}")
        
        # 注意：SDK通常不直接提供畸变系数，因为它们可能为零或需要通过标定获得。
        print("\n注意: Orbbec SDK 通常不直接提供畸变系数。")
        print("="*40)

    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        # 5. 确保资源被释放
        if pipeline:
            # 即使没有start(), stop()也是安全的
            pipeline.stop()
            print("\n程序结束，资源已释放。")



if __name__ == "__main__":
    get_and_print_factory_intrinsics()
