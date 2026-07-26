from challenge_images.category_map import class_to_mid, normalize_dataset_class


def test_aliases():
    assert normalize_dataset_class("traffic-lights") == "Traffic Light"
    assert normalize_dataset_class("fire_hydrant") == "Hydrant"
    assert class_to_mid("红绿灯") == "/m/015qff"
