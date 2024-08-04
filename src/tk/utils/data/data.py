import torch
from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    """Dataloader that reuses workers, ref [yolov3].

    See some examples of similar classes [0,1].
    
    [yolov3]: https://github.com/ultralytics/yolov3/blob/3a9231af38391c3f879b85bcc61ab813058c096c/utils/dataloaders.py#L185
    [0]: https://github.com/m1k2zoo/FedMedICL/blob/5fb71fa61077a47c0ea1a68724e406b08e7b9f80/training/util.py#L371
    [1]: https://github.com/DingXiaoH/ACNet/blob/748fb0c734b41c48eacaacf7fc5e851e33a63ce8/deprecated/dataset.py#L10
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
