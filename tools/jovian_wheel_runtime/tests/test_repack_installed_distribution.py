from pathlib import Path, PurePosixPath

from tools.jovian_wheel_runtime.repack_installed_distribution import (
    is_installation_generated,
    normalize_tree_mtime,
)


def test_excludes_only_installation_generated_wheel_entries():
    assert is_installation_generated(PurePosixPath("../../../bin/tool"))
    assert is_installation_generated(PurePosixPath("package/__pycache__/module.pyc"))
    assert is_installation_generated(PurePosixPath("package-1.0.dist-info/RECORD"))
    assert is_installation_generated(
        PurePosixPath("package-1.0.dist-info/direct_url.json")
    )
    assert not is_installation_generated(PurePosixPath("package/module.py"))
    assert not is_installation_generated(
        PurePosixPath("package-1.0.dist-info/licenses/LICENSE")
    )


def test_normalizes_staged_file_and_directory_mtimes(tmp_path: Path):
    directory = tmp_path / "package"
    directory.mkdir()
    payload = directory / "module.py"
    payload.write_text("VALUE = 1\n")

    normalize_tree_mtime(tmp_path, 1_700_000_000)

    assert int(tmp_path.stat().st_mtime) == 1_700_000_000
    assert int(directory.stat().st_mtime) == 1_700_000_000
    assert int(payload.stat().st_mtime) == 1_700_000_000
