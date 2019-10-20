from collections import namedtuple


Dataset = namedtuple("Dataset", ["data", "targets"])

TrainTestDataset = namedtuple(
    "TrainTestDataset",
    ["train", "test"]
)
