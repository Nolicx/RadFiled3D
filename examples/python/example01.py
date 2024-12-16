from RadFiled3D.RadFiled3D import CartesianRadiationField, FieldStore, vec3, DType, StoreVersion
from RadFiled3D.RadFiled3D import RadiationFieldMetadataV1, RadiationFieldSimulationMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSoftwareMetadataV1


## See the C++ example for more details on the API usage
## Also the C++ docstrings are more detailed

if __name__ == "__main__":
    # Create a CartesianRadiationField object with a size of 2m x 2m x 2m and a voxel size of 0.1m
    # Alternatively, you can work with a PolarRadiationField, if you adress your voxels in polar coordinates
    crf = CartesianRadiationField(
        vec3(2.0, 2.0, 2.0),
        vec3(0.1, 0.1, 0.1),
    )
    print(f"Field: {crf}")
    # Add a channel to the field
    channel = crf.add_channel("scattering")

    # Add layers to the channel that spanns voxels over the whole field
    # that each store one element of the given data type
    # The second argument is the unit of the data
    channel.add_layer("doserate", "Gy/h", DType.FLOAT32)
    channel.add_layer("energy", "Gy", DType.FLOAT64)
    channel.add_layer("hits", "counts", DType.UINT64)
    channel.add_layer("directions", "", DType.VEC3)

    print(channel.get_layers())

    # Access a voxel from a layer and set its data using explicit methods
    voxel = channel.get_voxel("doserate", 0, 0, 0)
    voxel.set_data(1.0)

    # Access a voxel from a layer using the pythonic way
    # Load an entire layer as a numpy array
    # Note that the data is not copied, so changes to the array will be reflected in the field
    array = channel.get_layer_as_ndarray("doserate")
    print("Doserate array:")
    print(f"Dtype: {array.dtype} shape: {array.shape}")
    print(f"Check updated data: {array[0, 0, 0]}")
    print(f"Max value: {array.max()}")
    print(f"Min value: {array.min()}")

    # You can even use slicing to update the data
    array[0, 1:, 0] = 2.0

    # Check the written data
    chk_voxel = channel.get_voxel("doserate", 0, 5, 0)
    print(f"Check updated data: {chk_voxel.get_data()}")

    # The conversion to numpy even works for more complex data types
    array = channel.get_layer_as_ndarray("directions")
    print("Directions array:")
    print(f"Dtype: {array.dtype} shape: {array.shape}")

    channel.get_voxel("directions", 0, 0, 0).set_data(vec3(1.67, 1.85, 2.0))
    voxel = channel.get_voxel("directions", 0, 0, 0)
    print(f"Set data: {voxel.get_data()}")
    voxel = channel.get_voxel("directions", 0, 1, 0)
    print(f"Unset data: {voxel.get_data()}")

    # Create the metadata object for this field
    metadata = RadiationFieldMetadataV1(
        RadiationFieldSimulationMetadataV1(
            100,                            # Primary particles
            "SomeGeometry",                 # Geometry
            "PhysList",                     # Used physics list
            RadiationFieldXRayTubeMetadataV1( # X-Ray tube metadata
                vec3(0.0, 0.0, 0.0),
                vec3(0.0, 0.0, 0.0),
                0.0,
                "SomeTubeID"
            )
        ),
        RadiationFieldSoftwareMetadataV1(    # Software metadata
            "SomeSoftware",
            "SomeVersion",
            "SomeRepository",
            "SomeCommitID",
            "SomeDOI"                       # Optional DOI
        )
    )

    # Store the field and metadata to a file using the file version 1
    FieldStore.store(crf, metadata, "example01.rf3", StoreVersion.V1)

    # Load the field and metadata from a file
    # Note, that the file version is automatically detected
    field2 = FieldStore.load("example01.rf3")
    # Print the field parameters
    print(field2)

    # Load the metadata only
    # Note, that the file version is automatically detected
    metadata2 = FieldStore.get_metadata("example01.rf3")
