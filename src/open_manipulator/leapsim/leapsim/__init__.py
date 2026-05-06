import omegaconf
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
#from leapsim.utils.reformat import omegaconf_to_dict

print(f"Using OmegaConf version: {omegaconf.__version__}")


def omegaconf_to_dict(config):
    """
    安全地将 OmegaConf 配置转换为原生 Python 字典
    处理所有插值类型，包括 resolve_default
    """
    # 方法 1: 使用 OmegaConf 内置转换 (推荐)
    try:
        # 尝试解析所有插值
        resolved_config = OmegaConf.resolve(config)
        # 转换为原生 Python 类型
        return OmegaConf.to_container(resolved_config, resolve=True)
    except Exception as e:
        print(f"Error converting config: {e}")
        # 方法 2: 回退到手动转换
        return _fallback_conversion(config)

def _fallback_conversion(node):
    """手动递归转换 OmegaConf 节点"""
    if OmegaConf.is_dict(node):
        return {k: _fallback_conversion(v) for k, v in node.items()}
    elif OmegaConf.is_list(node):
        return [_fallback_conversion(v) for v in node]
    elif OmegaConf.is_interpolation(node):
        try:
            # 尝试解析插值
            return OmegaConf.resolve(node)
        except:
            # 无法解析则返回原始值
            return str(node)
    else:
        # 基础类型直接返回
        return node


OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def make(
    seed: int, 
    task: str, 
    num_envs: int, 
    sim_device: str,
    rl_device: str,
    graphics_device_id: int = -1,
    headless: bool = False,
    multi_gpu: bool = False,
    virtual_screen_capture: bool = False,
    force_render: bool = True,
    cfg: DictConfig = None
): 
    from leapsim.utils.rlgames_utils import get_rlgames_env_creator
    # create hydra config if no config passed in
    if cfg is None:
        # reset current hydra config if already parsed (but not passed in here)
        if HydraConfig.initialized():
            task = HydraConfig.get().runtime.choices['task']
            hydra.core.global_hydra.GlobalHydra.instance().clear()

        with initialize(config_path="./cfg"):
            cfg = compose(config_name="config", overrides=[f"task={task}"])
            cfg_dict = omegaconf_to_dict(cfg.task)
            cfg_dict['env']['numEnvs'] = num_envs
    # reuse existing config
    else:
        cfg_dict = omegaconf_to_dict(cfg.task)

    create_rlgpu_env = get_rlgames_env_creator(
        seed=seed,
        task_config=cfg_dict,
        task_name=cfg_dict["name"],
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=headless,
        multi_gpu=multi_gpu,
        virtual_screen_capture=virtual_screen_capture,
        force_render=force_render,
    )
    return create_rlgpu_env()

