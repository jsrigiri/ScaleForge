from scaleforge.data import ShardedDataset


def test_dataset_loads():

    dataset = ShardedDataset("data")

    assert len(dataset) > 0

    x, y = dataset[0]

    assert len(x.shape) == 1
    assert y.item() in [0, 1]