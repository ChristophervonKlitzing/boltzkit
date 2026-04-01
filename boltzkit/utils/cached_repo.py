from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
import yaml
from pathlib import PurePosixPath
from boltzkit.utils.key_value_store import FileKV


def strip_repo_prefix(full_path: str, repo_root: str) -> str:
    """
    Returns the relative path to the given repo root.
    """
    full = PurePosixPath(full_path)
    root = PurePosixPath(repo_root)
    return str(full.relative_to(root))


class CachedRepo(ABC):
    """
    Abstract base class representing a cached repository.

    A CachedRepo provides a unified interface for interacting with
    remote repositories (e.g., Huggingface datasets or local directories)
    while caching files locally for efficient repeated access.

    Attributes:
        remote_uri (str): The URI or path of the remote repository.
        local_path (Path): The local directory where files are cached.
    """

    def __init__(self, remote_uri: str, local_repo_path: Path, lazy_load: bool):
        """
        Initialize a CachedRepo instance.

        Args:
            remote_uri (str): The remote repository URI or path.
            local_repo_path (Path): Local path where cached files will be stored.
            lazy_load (bool): If True, files are loaded on demand; if False, all files are loaded immediately.
        """
        super().__init__()
        self.__remote_uri = remote_uri
        self.__local_path = local_repo_path
        self.__lazy_load = lazy_load

    def post_init(self):
        self._key_value_store = FileKV(self.local_path / "cached_config.yaml")

        info_path = self.load_file("info.yaml")
        with open(info_path) as f:
            self._config: dict[str, Any] = yaml.safe_load(f)

        if not self.__lazy_load:
            self.load_all_files()

    @abstractmethod
    def load_file(self, relative_fpath: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def load_all_files(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_remote_files(self) -> list[str]:
        raise NotImplementedError

    def find_file(self, regex: str) -> list[str]:
        """
        Return all remote files matching the given regex pattern.

        Args:
            regex (str): Regular expression to match against file paths.

        Returns:
            List[str]: List of matching file paths (repo-relative).
        """
        pattern = re.compile(regex)
        return [path for path in self.list_remote_files() if pattern.search(path)]

    @property
    def config(self) -> dict[str, Any]:
        if not hasattr(self, "_config"):
            raise AttributeError(
                "The attribute _config could not be found, perhaps 'post_init' was not called."
            )
        return self._config.copy()

    @classmethod
    @abstractmethod
    def match_uri(cls, uri: str) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_name_from_uri(cls, uri: str) -> str:
        raise NotADirectoryError

    @property
    def remote_uri(self) -> str:
        return self.__remote_uri

    @property
    def local_path(self) -> Path:
        return self.__local_path

    def get_cached_key_value_store(self):
        return self._key_value_store


class HuggingfaceRepo(CachedRepo):
    def __init__(self, remote_uri, local_repo_path, lazy_load):
        super().__init__(remote_uri, local_repo_path, lazy_load)
        self._fs = HfFileSystem()
        self._ignore_patterns = [".gitattributes"]

        self.post_init()

    def load_file(self, relative_fpath):
        local_path = hf_hub_download(
            repo_id=self.remote_uri.replace("datasets/", ""),
            repo_type="dataset",
            local_dir=self.local_path,
            filename=relative_fpath,
        )
        return Path(local_path)

    def load_all_files(self):
        snapshot_download(
            repo_id=self.remote_uri.replace("datasets/", ""),
            repo_type="dataset",
            local_dir=self.local_path,
            ignore_patterns=self._ignore_patterns,
        )

    def list_remote_files(self):
        l = [
            strip_repo_prefix(p, self.remote_uri)
            for p in self._fs.find(self.remote_uri)
        ]
        # filter out unwanted files like .gitattributes
        return list(filter(lambda x: x not in self._ignore_patterns, l))

    @classmethod
    def match_uri(cls, uri):
        if not uri.startswith("datasets/"):
            return False

        fs = HfFileSystem()

        try:
            # Try listing the root of the repo
            fs.ls(uri)
            return True
        except Exception:
            return False

    @classmethod
    def get_name_from_uri(cls, uri):
        return uri.split("/")[-1]  # "huggingface_" +


class LocalRepo(CachedRepo):
    def __init__(self, remote_uri, local_repo_path, lazy_load):
        super().__init__(remote_uri, local_repo_path, lazy_load)
        self.post_init()

    def load_file(self, relative_fpath):
        remote_file = Path(self.remote_uri) / relative_fpath
        local_file = self.local_path / relative_fpath

        # create parent directories if needed
        local_file.parent.mkdir(parents=True, exist_ok=True)

        # create symlink if it doesn't exist
        if not local_file.exists():
            local_file.symlink_to(remote_file.resolve())

        return local_file

    def load_all_files(self):
        for remote_relative_fpath in self.list_remote_files():
            self.load_file(remote_relative_fpath)

    def list_remote_files(self):
        files = [
            str(p.relative_to(self.remote_uri))
            for p in Path(self.remote_uri).rglob("*")
            if p.is_file()
        ]
        return files

    @classmethod
    def match_uri(cls, uri):
        return Path(uri).expanduser().exists()

    @classmethod
    def get_name_from_uri(cls, uri):
        return "local_" + Path(uri).name


def create_cached_repo(
    uri: str,
    local_repos_dir: Path = Path("target_cache"),
    lazy_load: bool = True,
    **kwargs,
):
    """
    Creates CachedRepo object from the given URI (Unified Resource Identifier).
    The type of the CachedRepo is automatically determined by the given URI.
    """
    classes: list[type[CachedRepo]] = [HuggingfaceRepo, LocalRepo]

    for cls in classes:
        if cls.match_uri(uri):
            name = cls.get_name_from_uri(uri)
            local_repo_path = local_repos_dir / name
            local_repo_path.mkdir(parents=True, exist_ok=True)

            cached_repo = cls(
                remote_uri=uri,
                local_repo_path=local_repo_path,
                lazy_load=lazy_load,
                **kwargs,
            )
            print(
                f"Created cached repo of type '{cls.__name__}' for remote uri '{uri}' and local path '{local_repo_path.as_posix()}'"
            )
            return cached_repo


if __name__ == "__main__":
    repo_path = "datasets/chrklitz99/alanine_dipeptide"
    # repo_path = "target_cache/alanine_dipeptide"
    sys_info = create_cached_repo(repo_path)

    print("Remote files:")
    print(sys_info.list_remote_files())
    print(sys_info.config)

    print(sys_info.find_file(".*\.pdb"))
