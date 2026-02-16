from pathlib import Path
from typing import Any
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
import yaml
from pathlib import PurePosixPath


def is_local_path(path: str) -> bool:
    return Path(path).expanduser().exists()


def is_huggingface_path(path: str) -> bool:
    fs = HfFileSystem()

    try:
        # Try listing the root of the repo
        fs.ls(path)
        return True
    except Exception:
        return False


def strip_repo_prefix(full_path: str, repo_root: str) -> str:
    full = PurePosixPath(full_path)
    root = PurePosixPath(repo_root)
    return str(full.relative_to(root))


def detect_path_type(path: str):
    if is_local_path(path):
        return "local"
    elif is_huggingface_path(path):
        return "huggingface"
    else:
        return None


class CachedRepo:
    def __init__(
        self, path: str, local_cache_dir: str = "target_cache", lazy_download=True
    ):
        self._path = path.rstrip("/")
        self._repo_name = self._path.split("/")[-1]
        self._local_cache_dir = Path(local_cache_dir)
        self._lazy_download = lazy_download

        self._fs = HfFileSystem()

        # Create cache directory
        self._local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load initial file
        self._info_path = self.load_file("info.yaml")

        with open(self._info_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

        if not lazy_download:
            snapshot_download(
                repo_id=self._path.replace("datasets/", ""),
                repo_type="dataset",
                local_dir=self._local_cache_dir / self._repo_name,
                ignore_patterns=[".gitattributes"],
            )

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def list_remote_files(self):
        """Return all files in the HF repo."""
        return [strip_repo_prefix(p, self._path) for p in self._fs.find(self._path)]

    def load_file_old(self, relative_filepath: str):
        """
        Downloads a file from HF repo if not already cached.
        Returns local file path.
        """

        remote_path = f"{self._path}/{relative_filepath}"
        local_path = self._local_cache_dir / self._repo_name / relative_filepath

        # Create subdirectories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # If already cached → return
        if local_path.exists():
            return local_path

        # Check file exists remotely
        if not self._fs.exists(remote_path):
            raise FileNotFoundError(f"{remote_path} not found in HF repo.")

        # Download file
        self._fs.download(remote_path, str(local_path))

        return local_path

    def load_file(self, relative_filepath: str):
        repo_type, repo_id = self._path.split("/", 1)
        repo_type = repo_type[:-1]  # datasets -> dataset

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=relative_filepath,
            repo_type=repo_type,
            local_dir=self._local_cache_dir / self._repo_name,
        )

        return Path(local_path)


if __name__ == "__main__":
    repo_path = "datasets/chrklitz99/test_system"
    sys_info = CachedRepo(repo_path, lazy_download=True)

    print("Remote files:")
    print(sys_info.list_remote_files())
    print(sys_info.config)
