from challenge_images.annotation_store import AnnotationStore


def test_annotation_roundtrip(tmp_path):
    path = tmp_path / "annotations.json"
    store = AnnotationStore(path)
    store.set("image.jpg", challenge_type="dynamic", grid="3×3", target_class="Hydrant", indices=[6, 2, 6])
    loaded = AnnotationStore(path).get("image.jpg")
    assert loaded["真实格子"] == [2, 6]
