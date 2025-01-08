from RadFiled3D.RadFiled3D import FieldType, FieldStore, StoreVersion, CartesianRadiationField, DType, vec3, RadiationFieldMetadataV1, RadiationFieldSimulationMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSoftwareMetadataV1
import pickle


METADATA = RadiationFieldMetadataV1(
        RadiationFieldSimulationMetadataV1(
            100,
            "",
            "Phys",
            RadiationFieldXRayTubeMetadataV1(
                vec3(0, 0, 0),
                vec3(0, 0, 0),
                0,
                "TubeID"
            )
        ),
        RadiationFieldSoftwareMetadataV1(
            "RadFiled3D",
            "0.1.0",
            "repo",
            "commit"
        )
    )


def test_pickle():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test05.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test05.rf3")
    a_repr = repr(accessor)
    vx_count = accessor.get_voxel_count()
    true_vx_count = field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    b_repr = repr(accessor)
    assert a_repr == b_repr
    assert vx_count == true_vx_count
    assert accessor.get_field_type() == FieldType.CARTESIAN

    pickled_accessor = pickle.dumps(accessor)
    loaded_accessor = pickle.loads(pickled_accessor)

    b_repr = repr(loaded_accessor)
    assert loaded_accessor.get_voxel_count() == field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    assert loaded_accessor.get_field_type() == accessor.get_field_type()
    assert b_repr == a_repr


def test_accessing_field():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test06.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test06.rf3")
    field2 = accessor.access_field("test06.rf3")
    assert field2.get_voxel_counts() == field.get_voxel_counts()
    assert field2.get_channel("channel1").get_layer("layer1").get_dtype() == DType.FLOAT32


def test_accessing_layer():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    FieldStore.store(field, METADATA, "test07.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test07.rf3")
    layer = accessor.access_layer("channel1", "layer1")
    assert layer.get_dtype() == DType.FLOAT32
    assert layer.get_unit() == "unit1"


def test_accessing_voxel():
    field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
    field.add_channel("channel1")
    field.get_channel("channel1").add_layer("layer1", "unit1", DType.FLOAT32)
    field.get_channel("channel1").get_layer_as_ndarray("layer1")[:, :, :] = 2.34
    FieldStore.store(field, METADATA, "test08.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test08.rf3")
    voxel = accessor.access_voxel(0, 0, 0)
    assert voxel.get_data() == 2.34
