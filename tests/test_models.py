import numpy as np
import pytest

torch = pytest.importorskip("torch")

from maize.autoencoder import ConvAutoencoder, LeafDataset, load_model, reconstruction_errors, save_model


def fake_images(n=5):
    return np.random.default_rng(0).integers(0, 255, (n, 128, 128, 3), dtype=np.uint8)


def test_autoencoder_shapes_and_range():
    x = torch.rand(2, 3, 128, 128)
    y = ConvAutoencoder()(x)
    assert y.shape == x.shape and 0.0 <= float(y.min()) and float(y.max()) <= 1.0


def test_dataset_and_errors_keep_order():
    imgs = fake_images()
    assert LeafDataset(imgs)[0].shape == (3, 128, 128)
    errs = reconstruction_errors(ConvAutoencoder().eval(), imgs, "cpu", batch_size=2)
    assert errs.shape == (5,) and (errs >= 0).all()
    # same image twice -> same error (order is preserved, no shuffling)
    twice = reconstruction_errors(ConvAutoencoder().eval(), imgs[[0, 0]], "cpu")
    assert twice[0] == pytest.approx(twice[1])


def test_save_and_load_roundtrip(tmp_path):
    m = ConvAutoencoder()
    save_model(m, tmp_path / "m", {"data_version": "abc"})
    m2 = load_model(tmp_path / "m", "cpu")
    x = torch.rand(1, 3, 128, 128)
    assert torch.allclose(m.eval()(x), m2(x), atol=1e-6)


def test_onnx_export_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from maize.autoencoder import export_onnx
    m = ConvAutoencoder().eval()
    export_onnx(m, tmp_path / "model.onnx")
    sess = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
    x = torch.rand(2, 3, 128, 128)
    out = sess.run(None, {"image": x.numpy()})[0]            # batch of 2: dynamic batch axis works
    assert out.shape == (2, 3, 128, 128)
    assert np.allclose(out, m(x).detach().numpy(), atol=1e-4)
