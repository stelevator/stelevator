import importlib.metadata
from .emulators import MESASolarLikeEmulator

__version__ = importlib.metadata.version(__package__)  # only works if package installed via pip
