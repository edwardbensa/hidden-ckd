# src/import_helper.py
import importlib.util
from pathlib import Path

def import_config():
    """Import config module from file path"""
    config_path = Path(__file__).parent / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

config = import_config()