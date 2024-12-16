# RadFiled3D
This Repository holds the Fileformat and API according to the Paper: "RadField3D: A Data Generator and Data Format for Deep Learning in Radiation-Protection Dosimetry for Medical Applications".

The aim of this library is, to provide a simple to use API for a structured, binary file format, that can store all relevant information from a three dimensional radiation field calculated by applications that use algorithms like Monte-Carlo radiation transport simulations. Such a binary file format is useful, when one needs to process a huge amount of radiation field files like when training a neural network. With that use-case in mind, RadFiled3D also provides a python interface with a pyTorch integration.

## Building and Installing
You can build and install this library and python module from source by using CMake and a C++ compiler. The CMake Project will be built automatically, but will take some time.

### Prequisites
- C++ Compiler (Tested with MSVC for Windows and g++ for Linux)
- CMake >= 3.30
- Python >= 3.11

### CMake
In order to use the module directly from another C++ Project, you can integrate it by adding the local location of this repository via `add_submodule()` and then link against the target `libRadFiled3D`. All classes are then available from the namespace `RadFiled3D`. Check the [Example](./examples/cxx/example01.cpp) or the [First Test File](./tests/basic.cpp) as a first reference.

### Python
In order to use the Module from Python, we provide a setup.py file that handles the compilation and integration automatically from the python setuptools.
#### Installing locally
`python -m pip install .`

#### Building a wheel
`python -m build --wheel`

## Getting Started
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