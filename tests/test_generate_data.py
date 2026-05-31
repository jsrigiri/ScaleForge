from scaleforge.generate_data import generate_record


def test_generate_record():
    record = generate_record(
        seq_len=8,
        vocab_size=32,
    )

    assert "tokens" in record
    assert "label" in record
    assert len(record["tokens"]) == 8
    assert record["label"] in [0, 1]
    assert all(
        1 <= token < 32
        for token in record["tokens"]
    )