from scaleforge.config import load_config


def test_load_config():
    config = load_config("configs/config.yaml")

    assert config["project"]["name"] == "ScaleForge"
    assert "training" in config
    assert "data" in config
    assert "distributed" in config