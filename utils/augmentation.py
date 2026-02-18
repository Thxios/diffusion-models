

import torchvision.transforms as T
from typing import Optional, List


AUGUMENTATION_TYPE_MAPPING = {
    'RandomHorizontalFlip': T.RandomHorizontalFlip,
    'RandomVerticalFlip': T.RandomVerticalFlip,
}

def get_augmentations(augmentations: Optional[List[str]] = None):
    transform = []
    if augmentations is not None:
        for aug_type in augmentations:
            assert aug_type in AUGUMENTATION_TYPE_MAPPING, f'unknown aug type {aug_type}'
            transform.append(AUGUMENTATION_TYPE_MAPPING[aug_type]())
    return transform
