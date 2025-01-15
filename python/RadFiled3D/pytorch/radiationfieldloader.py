from RadFiled3D.RadFiled3D import uvec2, vec2, uvec3, vec3, FieldStore, RadiationField, PolarRadiationField, CartesianRadiationField, RadiationFieldMetadata, VoxelGrid, PolarSegments, FieldAccessor, CartesianFieldAccessor, PolarFieldAccessor, Voxel
import zipfile
from enum import Enum
from torch import Tensor
import os
from torch.utils.data import Dataset, random_split, DataLoader
from typing import Type, Union, Tuple, Callable
from pathlib import Path
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn, TaskProgressColumn, SpinnerColumn, MofNCompleteColumn
from rich import print


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
        :param metadata_load_mode: Mode for loading metadata. FULL loads the full metadata, HEADER only loads the header, DISABLED does not load metadata. Default is HEADER. The provided metdata is a RadiationFieldMetadata object or None.
        """
        if isinstance(file_paths, str) or isinstance(file_paths, Path):
            file_paths = [file_paths]

        if file_paths is not None:
            file_paths = [str(p) for p in file_paths]
        self.file_paths = file_paths
        self.zip_file = zip_file
        self.metadata_load_mode = metadata_load_mode
        if self.file_paths is None and self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                self.file_paths = [f for f in zip_ref.namelist() if f.endswith(".rf3")]
        elif self.file_paths is None and self.zip_file is None:
            raise ValueError("Either file_paths or zip_file must be provided.")
        
        assert isinstance(self.file_paths, list), "file_paths must be a list of file paths."
        self._field_accessor: FieldAccessor = None

    def _get_field_accessor(self) -> Union[FieldAccessor, CartesianFieldAccessor, PolarFieldAccessor]:
        if self._field_accessor is None:
            if self.is_dataset_zipped:
                self._field_accessor = FieldStore.construct_field_accessor_from_buffer(self.load_file_buffer(0))
            else:
                self._field_accessor = FieldStore.construct_field_accessor(self.file_paths[0])
        return self._field_accessor
    
    field_accessor: Union[FieldAccessor, CartesianFieldAccessor, PolarFieldAccessor] = property(_get_field_accessor)
    is_dataset_zipped: bool = property(lambda self: self.zip_file is not None)

    def __len__(self):
        return len(self.file_paths)

    def load_file_buffer(self, idx: int) -> bytes:
        """
        Loads a binary file buffer from the dataset given an file index.
        :param idx: The index of the file in the dataset.
        :return: The binary file buffer.
        """
        if self.zip_file is not None:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(self.file_paths[idx]) as file:
                    return file.read()
        else:
            return open(self.file_paths[idx], 'rb').read()
    
    def _get_field(self, idx: int) -> Union[RadiationField, CartesianRadiationField, PolarRadiationField]:
        """
        Loads a radiation field from the dataset given a file index.
        :param idx: The index of the file in the dataset.
        :return: The radiation field.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_field_from_buffer(self.load_file_buffer(idx))
        else:
            return self.field_accessor.access_field(self.file_paths[idx])

    def check_dataset_integrity(self) -> bool:
        """
        Checks if all radiation field files in the dataset are valid.
        :return: True, if all files are valid, False otherwise.
        """
        valid = True
        invalid_files_count = 0
        progressbar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            MofNCompleteColumn()
        )
        with progressbar as progress:
            task = progress.add_task("Checking dataset integrity...", total=len(self.file_paths))
            for idx in range(len(self.file_paths)):
                try:
                    field = self._get_field(idx)
                    if field is None:
                        raise ValueError("Field is None.")
                except Exception as e:
                    valid = False
                    print(f"Error loading file {self.file_paths[idx]}: {str(e)}")
                    invalid_files_count += 1
                progress.update(task, advance=1)
        if not valid:
            print(f"Dataset contains {invalid_files_count} invalid files.")
        return valid
    
    def _get_metadata(self, idx: int) -> Union[RadiationFieldMetadata, None]:
        """
        Loads the metadata of a radiation field from the dataset given a file index. This method respects the metadata_load_mode.
        :param idx: The index of the file in the dataset.
        :return: The metadata of the radiation field.
        """
        if self.is_dataset_zipped:
            file_buffer = self.load_file_buffer(idx)
            if self.metadata_load_mode == MetadataLoadMode.FULL:
                metadata: RadiationFieldMetadata = FieldStore.peek_metadata_from_buffer(file_buffer)
            elif self.metadata_load_mode == MetadataLoadMode.HEADER:
                metadata: RadiationFieldMetadata = FieldStore.peek_metadata_from_buffer(file_buffer)
            else:
                metadata = None
        else:
            file_path = self.file_paths[idx]
            if self.metadata_load_mode == MetadataLoadMode.FULL:
                metadata: RadiationFieldMetadata = FieldStore.load_metadata(file_path)
            elif self.metadata_load_mode == MetadataLoadMode.HEADER:
                metadata: RadiationFieldMetadata = FieldStore.peek_metadata(file_path)
            else:
                metadata = None
        return metadata

    def _get_voxel_flat(self, file_idx: int, vx_idx: int, channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel index.
        :param file_idx: The index of the file in the dataset.
        :param vx_idx: The index of the voxel in the radiation field.
        :param channel_name: The name of the channel to load.
        :param layer_name: The name of the layer to load.
        :return: The voxel.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel_flat_from_buffer(self.load_file_buffer(file_idx), channel_name, layer_name, vx_idx)
        else:
            return self.field_accessor.access_voxel_flat(self.file_paths[file_idx], channel_name, layer_name, vx_idx)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Union[Tensor, None]]:
        """
        Loads a radiation field and its metadata from the dataset given a file index.
        :param idx: The index of the file in the dataset.
        :return: A tuple containing the radiation field and its metadata. Format: (target_field, origin_metadata)
        """
        field = self._get_field(idx)
        metadata = self._get_metadata(idx)
        return (self.transform(field, idx), self.transform_origin(metadata, idx))

    def __getitems__(self, indices: list[int]) -> list[Tuple[RadiationField, Union[RadiationFieldMetadata, None]]]:
        return [self.__getitem__(idx) for idx in indices]

    def __iter__(self):
        return RadiationFieldDatasetIterator(self)

    def transform(self, field: Union[RadiationField, VoxelGrid, PolarSegments, Voxel], idx: int) -> Union[RadiationField, VoxelGrid, PolarSegments, Voxel, Tensor]:
        """
        Override to transform a RadFiled3D type into a torch tensor.
        This should be used as the target for the model.
        By default this just returns the original RadFiled3D type.
        :param field: The original RadFiled3D type.
        :param idx: The index of the element in the dataset.
        :return: The transformed RadFiled3D type.
        """
        return field

    def transform_origin(self, metadata: RadiationFieldMetadata, idx: int) -> Union[RadiationFieldMetadata, Tensor]:
        """
        Override to transform a RadiationFieldMetadata into a torch tensor.
        This should be used as the input for the model.
        By default this just returns the original RadiationFieldMetadata.
        :param metadata: The RadiationFieldMetadata to transform.
        :param idx: The index of the metadata in the dataset.
        :return: The transformed RadiationFieldMetadata.
        """
        return metadata


class RadiationFieldDatasetIterator:
    def __init__(self, dataset: RadiationFieldDataset):
        self.dataset = dataset
        self.idx = 0

    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[Union[RadiationField, Tensor], Union[RadiationFieldMetadata, None, Tensor]]:
        if self.idx < len(self.dataset):
            item = self.dataset[self.idx]
            self.idx += 1
            return item
        else:
            raise StopIteration


class CartesianFieldDataset(RadiationFieldDataset):
    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        field = self._get_field(0)
        assert isinstance(field, CartesianRadiationField), "Dataset must contain CartesianRadiationFields."

    def _get_field(self, idx: int) -> CartesianRadiationField:
        return super()._get_field(idx)
    
    def _get_field_accessor(self) -> CartesianFieldAccessor:
        return super()._get_field_accessor()
    
    field_accessor: CartesianFieldAccessor = property(_get_field_accessor)

    def _get_channel(self, idx: int, channel_name: str) -> VoxelGrid:
        """
        Loads a radiation channel from the dataset given a file index and a channel name.
        :param idx: The index of the file in the dataset.
        :param channel_name: The name of the channel to load.
        :return: The radiation channel.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_channel_from_buffer(self.load_file_buffer(idx), channel_name)
        else:
            return self.field_accessor.access_channel(self.file_paths[idx], channel_name)
        
    def _get_layer(self, idx: int, channel_name: str, layer_name: str) -> VoxelGrid:
        """
        Loads a radiation layer from the dataset given a file index, a channel name and a layer name.
        :param idx: The index of the file in the dataset.
        :param channel_name: The name of the channel to load.
        :param layer_name: The name of the layer to load.
        :return: The radiation layer.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_layer_from_buffer(self.load_file_buffer(idx), channel_name, layer_name)
        else:
            return self.field_accessor.access_layer(self.file_paths[idx], channel_name, layer_name)
  
    def _get_voxel(self, file_idx: int, vx_coord: Union[Tuple[int, int, int], uvec3], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel coordinate.
        :param file_idx: The index of the file in the dataset.
        :param vx_coord: The coordinate of the voxel in the radiation field in quantized space.
        :return: The voxel.
        """
        vx_coord = vx_coord if isinstance(vx_coord, uvec3) else uvec3(vx_coord[0], vx_coord[1], vx_coord[2])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel(self.load_file_buffer(file_idx), channel_name, layer_name, vx_coord)
        else:
            return self.field_accessor.access_voxel(self.file_paths[file_idx], channel_name, layer_name, vx_coord)
        
    def _get_voxel_by_coord(self, file_idx: int, vx_coord: Union[Tuple[float, float, float], vec3], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel coordinate.
        :param file_idx: The index of the file in the dataset.
        :param vx_coord: The coordinate of the voxel in the radiation field in world space.
        :return: The voxel.
        """
        vx_coord = vx_coord if isinstance(vx_coord, vec3) else vec3(vx_coord[0], vx_coord[1], vx_coord[2])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel_by_coord(self.load_file_buffer(file_idx), channel_name, layer_name, vx_coord)
        else:
            return self.field_accessor.access_voxel_by_coord(self.file_paths[file_idx], channel_name, layer_name, vx_coord)


class PolarFieldDataset(RadiationFieldDataset):
    def __init__(self, file_paths = None, zip_file = None, metadata_load_mode = MetadataLoadMode.HEADER):
        super().__init__(file_paths, zip_file, metadata_load_mode)
        field = self._get_field(0)
        assert isinstance(field, PolarRadiationField), "Dataset must contain PolarRadiationFields."

    def _get_field_accessor(self) -> PolarFieldAccessor:
        return super()._get_field_accessor()
    
    field_accessor: PolarFieldAccessor = property(_get_field_accessor)

    def _get_field(self, idx: int) -> PolarRadiationField:
        return super()._get_field(idx)
    
    def _get_layer(self, idx: int, channel_name: str, layer_name: str) -> PolarSegments:
        """
        Loads a radiation layer from the dataset given a file index, a channel name and a layer name.
        :param idx: The index of the file in the dataset.
        :param channel_name: The name of the channel to load.
        :param layer_name: The name of the layer to load.
        :return: The radiation layer.
        """
        if self.is_dataset_zipped:
            return self.field_accessor.access_layer_from_buffer(self.load_file_buffer(idx), channel_name, layer_name)
        else:
            return self.field_accessor.access_layer(self.file_paths[idx], channel_name, layer_name)
    
    def _get_voxel(self, file_idx: int, vx_idx: Union[Tuple[int, int], uvec2], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel index.
        :param file_idx: The index of the file in the dataset.
        :param vx_idx: The index of the voxel in the radiation field.
        :return: The voxel.
        """
        vx_idx = vx_idx if isinstance(vx_idx, uvec2) else uvec2(vx_idx[0], vx_idx[1])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel(self.load_file_buffer(file_idx), channel_name, layer_name, vx_idx)
        else:
            return self.field_accessor.access_voxel(self.file_paths[file_idx], channel_name, layer_name, vx_idx)

    def _get_voxel_by_coord(self, file_idx: int, vx_coord: Union[Tuple[float, float], vec2], channel_name: str, layer_name: str) -> Voxel:
        """
        Loads a voxel from the dataset given a file index and a voxel coordinate.
        :param file_idx: The index of the file in the dataset.
        :param vx_coord: The coordinate of the voxel in the radiation field in world space.
        :return: The voxel.
        """
        vx_coord = vx_coord if isinstance(vx_coord, vec2) else vec2(vx_coord[0], vx_coord[1])
        if self.is_dataset_zipped:
            return self.field_accessor.access_voxel_by_coord(self.load_file_buffer(file_idx), channel_name, layer_name, vx_coord)
        else:
            return self.field_accessor.access_voxel_by_coord(self.file_paths[file_idx], channel_name, layer_name, vx_coord)


class CartesianFieldSingleLayerDataset(CartesianFieldDataset):
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

    def __getitem__(self, idx: int) -> Tuple[VoxelGrid, Union[RadiationFieldMetadata, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        layer = self._get_layer(idx, self.channel_name, self.layer_name)
        metadata = self._get_metadata(idx)
        return (self.transform(layer, idx), self.transform_origin(metadata, idx))


class PolarFieldSingleLayerDataset(PolarFieldDataset):
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

    def __getitem__(self, idx: int) -> Tuple[PolarSegments, Union[RadiationFieldMetadata, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        layer = self._get_layer(idx, self.channel_name, self.layer_name)
        metadata = self._get_metadata(idx)
        return (self.transform(layer, idx), self.transform_origin(metadata, idx))


class CartesianSingleVoxelDataset(CartesianFieldSingleLayerDataset):
    def __init__(self, file_paths: list[str] = None, zip_file: str = None, metadata_load_mode: MetadataLoadMode = MetadataLoadMode.HEADER):
        super().__init__(file_paths=file_paths, zip_file=zip_file, metadata_load_mode=metadata_load_mode)
        field = self._get_field(0)
        self.field_voxel_counts = field.get_voxel_counts()
        self.voxels_per_field = self.field_voxel_counts.x * self.field_voxel_counts.y * self.field_voxel_counts.z
        self.zip_ref = None # remove zip reference to avoid pickling issues
        self._field_accessor = None # remove field accessor to avoid pickling issues

    def __len__(self) -> int:
        vx_count = int(self.field_accessor.get_voxel_count())
        self._field_accessor = None  # remove field accessor to avoid pickling issues
        return super().__len__() * vx_count
    
    def _get_field(self, idx: int) -> CartesianRadiationField:
        return super()._get_field(idx // self.field_accessor.get_voxel_count())
    
    def _get_metadata(self, idx) -> Union[RadiationFieldMetadata, None]:
        return super()._get_metadata(idx // self.field_accessor.get_voxel_count())

    def __getitem__(self, idx) -> Tuple[Tensor, Union[Tensor, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        vx_idx = idx % self.field_accessor.get_voxel_count()
        ret_val = (self._get_voxel_flat(idx // self.field_accessor.get_voxel_count(), vx_idx, self.channel_name, self.layer_name), self._get_metadata(idx // self.field_accessor.get_voxel_count()))
        return (self.transform(ret_val[0], idx), self.transform_origin(ret_val[1], idx))


class PolarSingleVoxelDataset(PolarFieldSingleLayerDataset):
    def __init__(self, file_paths = None, zip_file = None, metadata_load_mode = MetadataLoadMode.HEADER):
        super().__init__(file_paths, zip_file, metadata_load_mode)
        field = self._get_field(0)
        self.field_voxel_counts = field.get_voxel_counts()
        self.voxels_per_field = self.field_voxel_counts.x * self.field_voxel_counts.y
        self.zip_ref = None # remove zip reference to avoid pickling issues
        self._field_accessor = None # remove field accessor to avoid pickling issues

    def __len__(self) -> int:
        vx_count = int(self.field_accessor.get_voxel_count())
        self._field_accessor = None  # remove field accessor to avoid pickling issues
        return super().__len__() * vx_count
    
    def _get_field(self, idx: int) -> PolarRadiationField:
        return super()._get_field(idx // self.field_accessor.get_voxel_count())
    
    def _get_metadata(self, idx) -> Union[RadiationFieldMetadata, None]:
        return super()._get_metadata(idx // self.field_accessor.get_voxel_count())

    def __getitem__(self, idx) -> Tuple[Tensor, Union[Tensor, None]]:
        assert self.channel_name is not None and self.layer_name is not None, "Channel and layer must be set before loading the radiation field."
        vx_idx = idx % self.field_accessor.get_voxel_count()
        ret_val = (self._get_voxel_flat(idx // self.field_accessor.get_voxel_count(), vx_idx, self.channel_name, self.layer_name), self._get_metadata(idx // self.field_accessor.get_voxel_count()))
        return (self.transform(ret_val[0], idx), self.transform_origin(ret_val[1], idx))


class DataLoaderBuilder(object):
    """
    A class that builds a RadiationFieldDataset from a directory or zip file and constructs DataLoaders for training, validation and testing.
    The dataset is split into train, validation and test sets according to the ratios provided in the constructor.
    Please note, that when using custom dataset classes to inherit from RadiationFieldDataset, the class must be provided as a parameter to the constructor.
    When using a custom dataset class, only the arguments file_paths and zip_file are passed to the constructor.
    """ 

    def __init__(self, dataset_path: str, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, dataset_class: Type[RadiationFieldDataset] = RadiationFieldDataset, on_dataset_created: Callable[[RadiationFieldDataset], None] = None):
        """
        :param dataset_path: The path to the directory or zip file containing the radiation field files.
        :param train_ratio: The ratio of the dataset that is used for training. Default is 0.7.
        :param val_ratio: The ratio of the dataset that is used for validation. Default is 0.15.
        :param test_ratio: The ratio of the dataset that is used for testing. Default is 0.15.
        :param dataset_class: The class of the dataset. Default is RadiationFieldDataset.
        :param on_dataset_created: A callback function that is called when the dataset is created. The function is called with the dataset as parameter.
        """
        self.on_dataset_created = on_dataset_created
        self.dataset_path = dataset_path
        self.dataset_path = str(self.dataset_path)
        self.dataset_class = dataset_class
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        if os.path.isdir(dataset_path):
            self.zip_file = None
            self.file_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".rf3")]
            if len(self.file_paths) == 0 and os.path.isdir(os.path.join(dataset_path, "fields")):
                self.file_paths = [os.path.join(dataset_path, "fields", f) for f in os.listdir(os.path.join(dataset_path, "fields")) if f.endswith(".rf3")]
            elif len(self.file_paths) == 0:
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
        if self.on_dataset_created is not None:
            self.on_dataset_created(ds)
        return ds
    
    def build_val_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.val_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        if self.on_dataset_created is not None:
            self.on_dataset_created(ds)
        return ds
    
    def build_test_dataset(self) -> RadiationFieldDataset:
        ds = self.dataset_class(file_paths=self.test_files, zip_file=self.zip_file)
        assert isinstance(ds, RadiationFieldDataset), "dataset_class was not related to RadiationFieldDataset"
        if self.on_dataset_created is not None:
            self.on_dataset_created(ds)
        return ds

    def build_dataloader(self, dataset: RadiationFieldDataset, batch_size=1, shuffle=True, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the dataset.
        :param dataset: The dataset to build the DataLoader for.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        is_multiprocessing = worker_count is None or worker_count != 0
        if worker_count is None or worker_count < 0:
            worker_count = max(1, os.cpu_count() - 1)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=worker_count, pin_memory=is_multiprocessing, persistent_workers=is_multiprocessing)
        dl.dataset._field_accessor = None
        return dl

    def build_train_dataloader(self, batch_size=1, shuffle=True, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the training dataset.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        return self.build_dataloader(self.build_train_dataset(), batch_size=batch_size, shuffle=shuffle, worker_count=worker_count)
    
    def build_val_dataloader(self, batch_size=1, shuffle=False, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the validation dataset.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        return self.build_dataloader(self.build_val_dataset(), batch_size=batch_size, shuffle=shuffle, worker_count=worker_count)
    
    def build_test_dataloader(self, batch_size=1, shuffle=False, worker_count: Union[int, None] = 0):
        """
        Builds a DataLoader for the test dataset.
        :param batch_size: The batch size. Default is 1.
        :param shuffle: Whether to shuffle the dataset. Default is True.
        :param worker_count: The number of workers for multiprocessing. Default is 0. If None or < 0, the number of workers is set to the number of CPUs minus 1.
        """
        return self.build_dataloader(self.build_test_dataset(), batch_size=batch_size, shuffle=shuffle, worker_count=worker_count)
