from pathlib import Path

import pytest

from meviewer.manifest import load


def test_relative_data_root_resolves_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (tmp_path / "viewer.toml").write_text(
        "[inputs]\nmesh_root = 'meshes'\nactive_frames = 'active_frames.json'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    resolved = load(Path("viewer.toml"), Path("data"))

    assert resolved == {
        "mesh_root": (data_root / "meshes").resolve(),
        "active_frames": (data_root / "active_frames.json").resolve(),
    }


def test_manifest_rejects_paths_outside_data_root(tmp_path: Path) -> None:
    manifest = tmp_path / "viewer.toml"
    manifest.write_text(
        "[inputs]\nmesh_root = '../outside'\nactive_frames = 'active_frames.json'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes data root"):
        load(manifest, tmp_path / "data")
