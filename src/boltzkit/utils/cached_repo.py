from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any, Callable, TypeAlias, Union
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
import yaml
from pathlib import PurePosixPath
from boltzkit.utils.key_value_store import FileKV
from pathlib import PurePosixPath


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
        pattern = r"^datasets\/[a-zA-Z0-9._-]+(\/.*)?$"
        return bool(re.match(pattern, uri))

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


Content: TypeAlias = str | bytes | Callable[[Path], None]
"""
str: Write text
bytes: write binary
Callable: creates file at path
"""


def normalize_path(path: str | PurePosixPath) -> str:
    return str(PurePosixPath(path))


class VirtualRepo(CachedRepo):
    """
    Creates cache directory from in-memory content,
    i.e., cache dir is not backed by some form of directory or repository
    """

    def __init__(
        self,
        remote_uri,
        local_repo_path,
        lazy_load,
        file_tree: dict[str, Content],
    ):
        """
        remote_uri must have format 'virtual://<name>', e.g., 'virtual://foo',
        which will create a cache dir with name 'virtual_foo'.
        """
        super().__init__(remote_uri, local_repo_path, lazy_load)

        self._file_content_tree = {normalize_path(k): v for k, v in file_tree.items()}
        self.post_init()

    def load_file(self, relative_fpath):
        relative_fpath = normalize_path(relative_fpath)
        target_path = self.local_path / relative_fpath
        target_path.parent.mkdir(parents=True, exist_ok=True)

        content = self._file_content_tree[relative_fpath]

        if isinstance(content, str):
            target_path.write_text(content)
            return target_path

        if isinstance(content, bytes):
            target_path.write_bytes(content)
            return target_path

        # Callable case: responsible for writing the file itself
        content(target_path)

        # Optional safety check
        if not target_path.exists():
            raise RuntimeError(f"Callable did not create file: {relative_fpath}")

        return target_path

    def load_all_files(self):
        for path in self._file_content_tree.keys():
            self.load_file(path)

    def list_remote_files(self):
        return list(self._file_content_tree.keys())

    @classmethod
    def match_uri(cls, uri):
        return uri.startswith("virtual://")

    @classmethod
    def get_name_from_uri(cls, uri):
        return "virtual_" + uri.removeprefix("virtual://")


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
    classes: list[type[CachedRepo]] = [HuggingfaceRepo, LocalRepo, VirtualRepo]

    if isinstance(local_repos_dir, str):
        local_repos_dir = Path(local_repos_dir)

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

    virtual_repo = create_cached_repo(
        "virtual://test", file_tree={"data/test.yaml": "Hello world"}
    )
    virtual_repo.load_file("data/test.yaml")
    print(virtual_repo.list_remote_files())
