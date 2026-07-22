import os

from jetarm.config.paths import PROJECT_ROOT


DEBUG = os.environ.get("JETARM_DEBUG", "0") == "1"
DEVICE = os.environ.get("JETARM_DEVICE", "cpu")
LOG_LEVEL = os.environ.get("JETARM_LOG_LEVEL", "INFO")

