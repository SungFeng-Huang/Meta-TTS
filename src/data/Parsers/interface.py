from pathlib import Path
import abc


class BaseRawParser(metaclass=abc.ABCMeta):
    def __init__(self, root: Path, *args, **kwargs) -> None:
        self.root = root

    @abc.abstractmethod
    def prepare_initial_features(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def parse(self, *args, **kwargs):
        raise NotImplementedError


class BasePreprocessor(metaclass=abc.ABCMeta):
    def __init__(self, root: Path, *args, **kwargs) -> None:
        self.root = root

    @abc.abstractmethod
    def prepare_mfa(self, mfa_data_dir: Path) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def mfa(self, mfa_data_dir: Path) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def denoise(self, *args, **kwargs) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def preprocess(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def split_dataset(self, *args, **kwargs) -> None:
        raise NotImplementedError
