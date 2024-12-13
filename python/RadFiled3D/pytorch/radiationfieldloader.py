from RadFiled3D.RadFiled3D import FieldStore, RadiationField, RadiationFieldMetadataHeader, RadiationFieldMetadata, VoxelGrid, PolarSegments
import zipfile
from enum import Enum
import os
from torch.utils.data import Dataset, random_split
from typing import Type, Union, Tuple, Any
import h5py
from pathlib import Path
import pickle


class MetadataLoadMode(Enum):
    FULL = 1
    HEADER = 2
    DISABLED = 3


class RadiationFieldDataset(Dataset):
    """
    A dataset that loads radiation field files and returns them as (field, metadata)-tuples.
    The dataset can be initialized with either a list of file paths in the file system (uncompressed) or a path to a zip file containing radiation field files.
    In the latter case, the file paths are either extracted from the zip file or can be provided as a list of relative paths. This is encouraged, as the splitting of the dataset in train, validation and test should be random an therefore all file paths should be known at the time of initialization.

    The dataset can be created by using the DatasetBuilder class. This allows the Builder to parse the zip or folder structure correctly and link the metadata to the radiation field files.

    The dataset is designed to be used with a DataLoader. The DataLoader should be initialized with a batch size of 1, as the radiation field files are already stored in memory and the dataset is not designed to be used with multiprocessing.
    """

    def __init__(self, file_paths: Union[list[str], str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        """
        :param file_paths: List of file paths to radiation field files. If zip_file is provided, this parameter can be None. In this case the file paths are extracted from the zip file. If file_paths is str, then it will be checked if it is an hdf5 file. If so it will be treated as an preprocessed dataset and loaded as such.
        :param zip_file: Path to a zip file containing radiation field files. If file_paths is provided, this parameter can be None. In this case the file paths are extracted from the zip file.
        :param metadata_load_mode: Mode for loading metadata. FULL loads the full metadata, HEADER only loads the header, DISABLED does not load metadata. Default is HEADER.
        """
        #assert file_paths is None or isinstance(file_paths, list) or isinstance(file_paths, str), "file_paths must be None, a list of strings or a string."
        if isinstance(file_paths, Path):
            file_paths = str(file_paths)
        self.file_paths = file_paths
        self.is_preprocessed = file_paths is not None and isinstance(file_paths, str) and os.path.isfile(file_paths) and file_paths.lower().endswith(".hdf5")
        self.zip_file = zip_file
        self.metadata_load_mode = metadata_load_mode
        if self.file_paths is None and self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                self.file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
        elif self.file_paths is not None and self.zip_file is not None:
            pass
        else:
            raise ValueError("Either file_paths or zip_file must be provided. Best practice is to provide both.")

    def prepare_dataset(self):
        """
        Prepares the dataset by loading all radiation fields and metadata into memory.
        """
        import numpy as np

        try:
            import torch
        except ImportError:
            torch = None
            print("PyTorch is not installed. Some functionalities may not be available.")

        if len(self) == 0:
            return
        first_field, _ = self.__getitem__(0)
        if isinstance(first_field, np.ndarray) or (torch is not None and isinstance(first_field, torch.Tensor)):
            hdf5_path = os.path.join(os.path.dirname(self.zip_file), f"tmp/{os.path.splitext(os.path.basename(self.zip_file))[0]}_{self.channel_name}_{self.layer_name}.hdf5")
            if not os.path.exists(os.path.dirname(hdf5_path)):
                os.makedirs(os.path.dirname(hdf5_path))
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)
            with h5py.File(hdf5_path, 'w') as hdf5_file:
                for i, (field, metadata) in enumerate(self):
                    if isinstance(metadata, RadiationFieldMetadata):
                        metadata = metadata.get_header()
                    hdf5_file.create_dataset(f'field_{i}', data=field)
                    if metadata is not None:
                        hdf5_file.create_dataset(f'metadata_{i}', data=pickle.dumps(metadata))

    def __len__(self):
        return len(self.file_paths)

    def _get_radiation_field(self, idx: int) -> RadiationField:
        file_path = self.file_paths[idx]
        if self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(file_path) as file:
                    field: RadiationField = FieldStore.load_from_buffer(file.read())
        else:
            field: RadiationField = FieldStore.load(file_path)
        return field
    
    def _get_metadata(self, idx: int) -> Union[RadiationFieldMetadataHeader, RadiationFieldMetadata, None]:
        file_path = self.file_paths[idx]
        if self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(file_path) as file:
                    if self.metadata_load_mode == MetadataLoadMode.FULL:
                        metadata: RadiationFieldMetadata = FieldStore.peek_metadata_from_buffer(file.read())
                    elif self.metadata_load_mode == MetadataLoadMode.HEADER:
                        metadata: RadiationFieldMetadataHeader = FieldStore.peek_metadata_from_buffer(file.read())
                    else:
                        metadata = None
        else:
            if self.metadata_load_mode == MetadataLoadMode.FULL:
                metadata: RadiationFieldMetadata = FieldStore.load_metadata(file_path)
            elif self.metadata_load_mode == MetadataLoadMode.HEADER:
                metadata: RadiationFieldMetadataHeader = FieldStore.peek_metadata(file_path)
            else:
                metadata = None
        return metadata

    def __getitem__(self, idx: int) -> Tuple[RadiationField, Union[RadiationFieldMetadataHeader, RadiationFieldMetadata, None]]:
        if self.is_preprocessed:
            with h5py.File(self.file_paths, 'r') as hdf5_file:
                field = hdf5_file[f'field_{idx}']
                metadata = hdf5_file[f'metadata_{idx}']
                return (field, metadata)
        else:
            field = self._get_radiation_field(idx)
            metadata = pickle.loads(self._get_metadata(idx))
            return (self.transform_field(field), metadata)
    
    def __getitems__(self, indices: list[int]) -> list[Tuple[RadiationField, Union[RadiationFieldMetadataHeader, RadiationFieldMetadata, None]]]:
        return [self.__getitem__(idx) for idx in indices]
    
    def __iter__(self):
        return RadiationFieldDatasetIterator(self)
    
    def transform_field(self, field: RadiationField) -> Union[RadiationField, Any]:
        """
        Transforms a RadiationField into a numpy array, torch tensor, VoxelGrid, PolarSegments or RadiationField. Just as needed for the training process.
        :param field: The RadiationField to transform.
        :return: The transformed RadiationField.
        """
        return field


class RadiationFieldDatasetIterator:
    def __init__(self, dataset: RadiationFieldDataset):
        self.dataset = dataset
        self.idx = 0

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[RadiationField, Union[RadiationFieldMetadataHeader, RadiationFieldMetadata, None]]:
        if self.idx < len(self.dataset):
            item = self.dataset[self.idx]
            self.idx += 1
            return item
        else:
            raise StopIteration


class CartesianFieldSingleLayerDataset(RadiationFieldDataset):
    """
    A dataset that loads single layers from a single channel of a radiation field as VoxelGrids.
    Useful, when only a single layer of a single channel is needed for training.
    """

    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        self.channel_name: str = None
        self.layer_name: str = None

    def set_channel_and_layer(self, channel_name: str, layer_name: str):
        self.channel_name = channel_name
        self.layer_name = layer_name

    def __getitem__(self, idx: int) -> Tuple[VoxelGrid, Union[RadiationFieldMetadataHeader, RadiationFieldMetadata, None]]:
        return super().__getitem__(idx)
    
    def _get_radiation_field(self, idx: int) -> VoxelGrid:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."

        file_path = self.file_paths[idx]
        if self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(file_path) as file:
                    field: VoxelGrid = FieldStore.load_single_grid_layer_from_buffer(file.read(), self.channel_name, self.layer_name)
        else:
            field: VoxelGrid = FieldStore.load_single_grid_layer(file_path, self.channel_name, self.layer_name)
        return field


class PolarFieldSingleLayerDataset(RadiationFieldDataset):
    """
    A dataset that loads single layers from a single channel of a radiation field as PolarSegments.
    Useful, when only a single layer of a single channel is needed for training.
    """

    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        self.channel_name: str = None
        self.layer_name: str = None

    def set_channel_and_layer(self, channel_name: str, layer_name: str):
        self.channel_name = channel_name
        self.layer_name = layer_name

    def __getitem__(self, idx: int) -> Tuple[PolarSegments, Union[RadiationFieldMetadataHeader, RadiationFieldMetadata, None]]:
        return super().__getitem__(idx)
    
    def _get_radiation_field(self, idx: int) -> PolarSegments:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."

        file_path = self.file_paths[idx]
        if self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(file_path) as file:
                    field: PolarSegments = FieldStore.load_single_polar_layer_from_buffer(file.read(), self.channel_name, self.layer_name)
        else:
            field: PolarSegments = FieldStore.load_single_polar_layer(file_path, self.channel_name, self.layer_name)
        return field


class DatasetBuilder(object):
    """
    A class that builds a RadiationFieldDataset from a directory or zip file.
    The dataset is split into train, validation and test sets according to the ratios provided in the constructor.
    Please note, that when using custom dataset classes to inherit from RadiationFieldDataset, the class must be provided as a parameter to the constructor.
    When using a custom dataset class, only the arguments file_paths and zip_file are passed to the constructor.
    """ 

    def __init__(self, dataset_path: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, dataset_class: Type[RadiationFieldDataset] = RadiationFieldDataset):
        self.dataset_path = dataset_path
        self.dataset_class = dataset_class
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        if os.path.isdir(dataset_path):
            self.zip_file = None
            self.file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".rf3")]
            if len(self.file_paths) == 0 and os.path.isdir(os.path.join(dataset_path, "fields")) and os.path.isdir(os.path.join(dataset_path, "spectra")):
                self.file_paths = [os.path.join(dataset_path, "fields", f) for f in os.listdir(os.path.join(dataset_path, "fields")) if f.endswith(".rf3")]
            else:
                raise FileNotFoundError(f"No radiation field files found in directory {dataset_path}.")
        elif os.path.isfile(dataset_path) and zipfile.is_zipfile(dataset_path):
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                self.file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
            self.zip_file = dataset_path
        else:
            raise ValueError(f"Dataset path {dataset_path} is neither a directory nor a zip file.")
        
        self.train_files, self.val_files, self.test_files = random_split(self.file_paths, [train_ratio, val_ratio, test_ratio])

    def build_train_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.train_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        return ds
    
    def build_val_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.val_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        return ds
    
    def build_test_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.test_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        return ds
