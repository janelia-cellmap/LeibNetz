from .attentive_scalenet import build_attentive_scale_net
from .scalenet import build_scale_net
from .unet import build_unet
from .local_learning import (
    convert_to_bio,
    convert_to_backprop,
    HebbsRule,
    KrotovsRule,
    OjasRule,
    GeometricConsistencyRule,
)

# from .resnet import build_resnet
