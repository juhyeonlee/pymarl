REGISTRY = {}

from .basic_controller import BasicMAC
from .global_controller import GlobalMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["global_mac"] = GlobalMAC