import os
from typing import Optional

import numpy as np
import h5py


class H5AppendFile:
    def __init__(self, filename: str, dataset: str = "array") -> None:
        """
        Initialize the object and associate it with an HDF5 file.

        Args:
            filename: path to the HDF5 file
            dataset: name of dataset inside the file
        """
        self.filename = filename
        self.dataset_name = dataset

        if os.path.isfile(filename):
            self.initialize()

    def __del__(self) -> None:
        if self.is_initialized:
            self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.__del__()

    def initialize(self, array: Optional[np.ndarray] = None) -> None:
        """
        Initialize or open the dataset.

        Args:
            array: array used to initialize file if it doesn't exist
        """

        self.fp = h5py.File(self.filename, "a")

        if self.dataset_name in self.fp:
            dset = self.fp[self.dataset_name]
            self.shape = dset.shape
            self.dtype = dset.dtype
        else:
            if array is None:
                raise ValueError(
                    "File does not contain dataset and no array provided to initialize."
                )

            self.dtype = array.dtype
            self.shape = array.shape

            maxshape = (None,) + array.shape[1:]

            self.fp.create_dataset(
                self.dataset_name,
                data=array,
                maxshape=maxshape,
                chunks=True,
            )

    def append(self, array: np.ndarray) -> None:
        """
        Append array along axis 0.
        """

        if not isinstance(array, np.ndarray):
            if not isinstance(array, list):
                raise TypeError(
                    f"Input has wrong type: expected np.ndarray, got {type(array)}"
                )
            array = np.array(array)

        if not hasattr(self, "fp"):
            self.initialize(array=array)
            return

        dset = self.fp[self.dataset_name]

        if array.dtype != self.dtype:
            raise TypeError(
                f"Array has wrong dtype: expected {self.dtype}, got {array.dtype}"
            )

        if len(array.shape) != len(self.shape):
            raise ValueError(
                f"Array has wrong number of dimensions: expected {len(self.shape)} != {len(array.shape)}"
            )

        if self.shape[1:] != array.shape[1:]:
            raise ValueError(
                f"Arrays must match outside zeroth dimension: expected {self.shape[1:]}, got {array.shape[1:]}"
            )

        old_size = dset.shape[0]
        new_size = old_size + array.shape[0]

        dset.resize(new_size, axis=0)
        dset[old_size:new_size] = array

        self.shape = dset.shape

    @property
    def is_initialized(self) -> bool:
        return hasattr(self, "fp")


if __name__ == "__main__":
    store = H5AppendFile("test.h5")
    store.append(np.ones((50, 2)))
    store.append(np.zeros((25, 2)))

    with h5py.File("test.h5") as f:
        dset = f["array"]
        np_dset = dset[:]
        print(np_dset.shape)
        # print(np_dset)
