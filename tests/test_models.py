import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from maize.autoencoder import ConvAutoencoder, LeafDataset, load_model, reconstruction_errors, save_model


def write_images(tmp_path, n=5):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), np.random.default_rng(i).integers(0, 255, (200, 150, 3), dtype=np.uint8))
        paths.append(str(p))
    return paths


def test_autoencoder_shapes_and_range():
    x = torch.rand(2, 3, 128, 128)
    y = ConvAutoencoder()(x)
    assert y.shape == x.shape and 0.0 <= float(y.min()) and float(y.max()) <= 1.0


def test_dataset_and_errors_align_with_paths(tmp_path):
    paths = write_images(tmp_path)
    assert LeafDataset(paths)[0].shape == (3, 128, 128)
    errs = reconstruction_errors(ConvAutoencoder().eval(), paths, "cpu", batch_size=2)
    assert errs.shape == (5,) and (errs >= 0).all()
    # same image twice -> same error (order is preserved, no shuffling)
    twice = reconstruction_errors(ConvAutoencoder().eval(), [paths[0], paths[0]], "cpu")
    assert twice[0] == pytest.approx(twice[1])


def test_save_and_load_roundtrip(tmp_path):
    m = ConvAutoencoder()
    save_model(m, tmp_path / "m", {"data_version": "abc"})
    m2 = load_model(tmp_path / "m", "cpu")
    x = torch.rand(1, 3, 128, 128)
    assert torch.allclose(m.eval()(x), m2(x), atol=1e-6)
