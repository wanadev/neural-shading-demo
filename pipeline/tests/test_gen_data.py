import numpy as np
import pytest

from pipeline.gen_data import save_atomic, shard_dir


class _Boom:
    def __array__(self, *a, **k):
        raise RuntimeError("killed mid-write")


def test_interrupted_write_leaves_no_shard(tmp_path):
    f = tmp_path / "view_0000.npz"
    with pytest.raises(RuntimeError):
        save_atomic(f, X=np.zeros((1000, 17)), Y=_Boom())
    assert not f.exists()  # a torn shard would be skipped on resume and break the merge


def test_save_atomic_roundtrip(tmp_path):
    f = tmp_path / "view_0000.npz"
    save_atomic(f, X=np.ones(3))
    assert np.load(f)["X"].tolist() == [1, 1, 1]


def test_shards_keyed_by_settings(tmp_path):
    out = tmp_path / "train.npz"
    assert shard_dir(out, 128, 1024) != shard_dir(out, 64, 1024) != shard_dir(out, 64, 256)
