from scaleforge.streaming_data import StreamingShardedDataset


def test_streaming_dataset_loads():
    dataset = StreamingShardedDataset("data")

    first_item = next(iter(dataset))

    x, y = first_item

    assert len(x.shape) == 1
    assert y.item() in [0, 1]