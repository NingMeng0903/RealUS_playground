from types import SimpleNamespace

import pytest

from perception.benchmark.mv4dhmr import _cached_dinov2_loader


def test_cached_dinov2_skips_network_and_keeps_other_repositories(tmp_path):
    cache = tmp_path / "hub/facebookresearch_dinov2_main"
    cache.mkdir(parents=True)
    (cache / "hubconf.py").write_text("# installed official source\n")
    calls = []

    def original(repo, model, **kwargs):
        calls.append((repo, model, kwargs))
        return "model"

    torch = SimpleNamespace(hub=SimpleNamespace(load=original))
    with _cached_dinov2_loader(torch, tmp_path):
        assert torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False) == "model"
        torch.hub.load("other/repo", "other_model")
    assert calls[0] == (str(cache), "dinov2_vitl14", {"pretrained": False, "source": "local"})
    assert calls[1] == ("other/repo", "other_model", {})
    assert torch.hub.load is original


def test_failed_constructor_restores_hub_loader(tmp_path):
    cache = tmp_path / "hub/facebookresearch_dinov2_main"
    cache.mkdir(parents=True)
    (cache / "hubconf.py").touch()

    def original(*args, **kwargs):
        raise RuntimeError("model construction failed")

    torch = SimpleNamespace(hub=SimpleNamespace(load=original))
    with pytest.raises(RuntimeError, match="construction failed"):
        with _cached_dinov2_loader(torch, tmp_path):
            torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    assert torch.hub.load is original
