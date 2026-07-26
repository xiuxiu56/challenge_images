from challenge_images.data.sample_manager import SampleManager


def test_unknown_directory_is_empty(tmp_path):
    manager = SampleManager(tmp_path, "dynamic")
    assert len(manager) == 0
