from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, RandRotate90

def get_transforms(augment=True):
    t = [EnsureChannelFirst(), NormalizeIntensity()]
    if augment:
        t.append(RandRotate90(prob=0.5, spatial_axes=(0,1)))
    return Compose(t)