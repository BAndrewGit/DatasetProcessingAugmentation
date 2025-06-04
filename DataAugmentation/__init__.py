from .base import BaseAugmentation
from .smote_tomek import SMOTETomekAugmentation
#from .gan_techniques import WCGANAugmentation

__all__ = [
    "BaseAugmentation",
    "SMOTETomekAugmentation",
    #"WCGANAugmentation"
]