from torchvision.datasets import ImageFolder
from typing import Callable,  Optional, Tuple, Any
import json
from numpy.random import randint
import numpy as np


class ImageFolderWithCaptions(ImageFolder):
    def __init__(self,
                 root: str,
                 # tokenizer: Any,
                 transform: Optional[Callable] = None,
                 # num_tokens: int = 8
                 ):
        super().__init__(
            root,
            transform=transform
        )

        # self.tokenizer = tokenizer
        # self.num_tokens = num_tokens

        with open("labels.json", 'r') as f:
            self.labels = json.load(f)['imagenet']

        self.templates = [
            "itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a {} in a video game.",
            "art of the {}.",
            "a photo of the small {}."
        ]

    def __getitem__(self, index:int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        tmp_index = randint(0, len(self.templates))
        caption = self.templates[tmp_index]
        caption = caption.format(self.labels[target])

        # caption = self.tokenizer(caption.format(self.labels[target]),
        #                          return_tensors='pt',
        #                          max_length=self.num_tokens,
        #                          padding='max_length',
        #                          truncation=True)
        # print(caption['input_ids'])
        # exit()

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, caption


class ConceptualCaptions(ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None
                 ):
        super().__init__(
            root,
            transform=transform
        )

        self.samples = np.load(f"{self.root}/captions.npy", allow_pickle=True)

        for i, (path, caption) in enumerate(self.samples):
            self.samples[i] = (f"{self.root}/train/{path}", caption)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # target = None
        path, caption = self.samples[index]

        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return sample, caption
