from egomimic.configs.config import Config
from egomimic.configs.base_config import config_factory, get_all_registered_configs

# note: these imports are needed to register these classes in the global config registry
from egomimic.configs.mimicplay_config import MimicPlayConfig
from egomimic.configs.act_config import ACTConfig
