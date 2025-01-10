# RadFiled3D

![Tests](https://github.com/Centrasis/RadFiled3D/actions/workflows/package-test-publish.yml/badge.svg)

This Repository holds the Fileformat and API according to the Paper: "RadField3D: A Data Generator and Data Format for Deep Learning in Radiation-Protection Dosimetry for Medical Applications".

The aim of this library is, to provide a simple to use API for a structured, binary file format, that can store all relevant information from a three dimensional radiation field calculated by applications that use algorithms like Monte-Carlo radiation transport simulations. Such a binary file format is useful, when one needs to process a huge amount of radiation field files like when training a neural network. With that use-case in mind, RadFiled3D also provides a python interface with a pyTorch integration.

## Building and Installing
### Installing from PyPi
Prebuilt versions of this module for python 3.11 and 3.12 for Windows and most Linuxsystems can be installed directly by using pip.

``pip install RadFiled3D``

### Installing from Source
You can build and install this library and python module from source by using CMake and a C++ compiler. The CMake Project will be 
built automatically, but will take some time.

#### Prerequisites
- C++ Compiler
  - g++ or clang for Linux
  - MSVC or clang from Visual Studio 2022 for Windows
- CMake >= 3.30
- Python >= 3.11

#### CMake
In order to use the module directly from another C++ Project, you can integrate it by adding the local location of this repository via `add_submodule()` and then link against the target `libRadFiled3D`. All classes are then available from the namespace `RadFiled3D`. Check the [Example](./examples/cxx/example01.cpp) or the [First Test File](./tests/basic.cpp) as a first reference.

#### Python
In order to use the Module from Python, we provide a setup.py file that handles the compilation and integration automatically from the python setuptools.
##### Installing locally
`python -m pip install .`

##### Building a wheel
`python -m build --wheel`

## Getting Started
## From Python
Simple example on how to create and store a radiation field. Find more in the example file: [Example](./examples/python/example01.py)
```python
from RadFiled3D.RadFiled3D import CartesianRadiationField, FieldStore, StoreVersion, DType

# Creating a cartesian radiation field
field = CartesianRadiationField(vec3(2.5, 2.5, 2.5), vec3(0.05, 0.05, 0.05))
# defining a channel and a layer on it
field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)

# accessing the voxels by using numpy arrays
array = field.get_channel("channel1").get_layer_as_ndarray("layer1")
assert array.shape == (50, 50, 50)
# modify voxels content by using numpy array as no data is copied, just referenced
array[2:5, 2:5, 2:5] = 2.0

# addressing a voxel by providing a point in space
voxel = field.get_channel("channel1").get_voxel_by_coord("layer1", 0.1, 2.4, 5)

# Store changes to a file
metadata = RadiationFieldMetadataV1(...)
FieldStore.store(field, metadata, "test01.rf3", StoreVersion.V1)

# load data
field2 = FieldStore.load("test01.rf3")
metadata2 = FieldStore.load_metadata("test01.rf3")
```

### Integrating with pyTorch
RadFiled3D comes with a submodule at `RadFiled3D.pytorch`. This module provides some dataset classes to support the usage. Datasets can be loaded from folders or .zip-Files.
```python
from RadFiled3D.pytorch import MetadataLoadMode, CartesianFieldSingleLayerDataset, DataLoaderBuilder
from RadFiled3D.pytorch.helpers import load_tensor_from_layer
from RadFiled3D.RadFiled3D import VoxelGrid
from torch import Tensor


# Extend one of the provided dataset classes to match the output to the current needs
# The argument type of 'layer' may vary depending on the dataset type between RadiationField (Whole field), VoxelGridBuffer (Channel), VoxelGrid (Layer) and Voxel (Single Voxel)
# The argument idx will contain the requested index from the dataset just in case someone wants to alter the return value based on it.
class MyLayerDataset(CartesianFieldSingleLayerDataset):
    def transform(self, layer: VoxelGrid, idx: int) -> Tensor:
        return load_tensor_from_layer(layer)    # transform the layers data

    def transform_origin(self, metadata: RadiationFieldMetadataV1, idx) -> Tensor:
        direction = metadata.get_header().simulation.tube.radiation_direction   # transform selected data from the header
        return torch.tensor([direction.x, direction.y, direction.z], dtype=torch.float32)

def finalize_dataset(dataset: MyLayerDataset)
    dataset.set_channel_and_layer("test_channel", "test_layer")
    dataset.metadata_load_mode = MetadataLoadMode.HEADER

# Pass the dataset class and other options to the DataLoaderBuilder
builder = DataLoaderBuilder(
    "./test_dataset.zip",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    dataset_class=MyLayerDataset,
    on_dataset_created=finalize_dataset     # Optional: provide a finalizer to perform configuration of the dataset once it was created by the builder
)

# Build the training dataset
train_dl = builder.build_train_dataloader(
    batch_size=8,
    shuffle=True,
    worker_count=4
)

# iterate over the dataset
for field, metadata in train_dl:
    pass
```

## From C++

Simple example on how to create and store a radiation field. Find more in the example file: [Example](./examples/cxx/example01.cpp)
```c++
#include <RadFiled3D/storage/RadiationFieldStore.hpp>
#include <RadFiled3D/RadiationField.hpp>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

void main() {
    auto field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f)); // field extents: 2.5 m x 2.5 m x 2.5 m and voxel extents: 5 cm x 5 cm x 5 cm

    auto metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
        // learn about the existing data fields from the example file in ./examples/cxx/examples01.cpp
    )

    FieldStore::store(field, metadata, "test_field.rf3", StoreVersion::V1);

    auto field2 = FieldStore::load("test_field.rf3");
}
```

## Field Structure
RadFiled3D defines a field structure, that provides the user with the possibility to first define in which kind of space he wants to operate. Therefore one can choose between `CartesianRadiationField` and `PolarRadiationField`.
- *CartesianRadiationField*: Segments a room defined by an extent of the room itself and each cuboid voxel into a set of voxels. Each voxel can be addressed by a 3D position (coordinate: x, y, z), a 3D index (number of the voxel in each dimension) or a flat 1D index.
- *PolarRadiationField*: Segements the surface of a unit sphere into surface segments. Each segment (voxel) can be addressed by a 2D position (coordinate: theta, phi), a 2D index (number of the segment in each dimension) or a flat 1D index.

Fields are then partitioned into channels (`VoxelGridBuffer`/`PolarSegmentsBuffer`). All channels share the same size and resolution. A channel is again partitioned into layers (`VoxelGrid`/`PolarSegment`). Each layer holds the actual voxel data and can be constructed from various data types (int, float, double, uint32_t, uint64_t, glm::vec2, glm::vec3, glm::vec4, N-D-Histogram). Additionally, a layer has a unit string assigned to it as well as a statistical uncertainty to perserve those information.

## Dependencies
RadFiled3D comes with a possibly low amount of dependencies. We integrated the OpenGL Math Library (GLM) just to provide those datatypes out of the box and as GLM is a head-only library we suspect no issues by doing so.

All C++ dependencies:
- [GLM](https://github.com/g-truc/glm)

All python dependencies:
- [PyBind11](https://github.com/pybind/pybind11)
- [rich](https://github.com/Textualize/rich)
- [numpy](https://numpy.org/)
- Optional:
  - [pyTorch](https://pytorch.org/)