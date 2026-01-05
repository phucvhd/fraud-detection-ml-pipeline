import yaml
from pathlib import Path
from typing import Dict, Any


def merge_configs(base_config: Dict[Any, Any], override_config: Dict[Any, Any]) -> Dict[Any, Any]:
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(base_config_path: str, override_config_path: str = None) -> Dict[Any, Any]:
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if override_config_path and Path(override_config_path).exists():
        with open(override_config_path, 'r') as f:
            override_config = yaml.safe_load(f)
        config = merge_configs(config, override_config)

    return config


def save_config(config: Dict[Any, Any], output_path: str):
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)