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
    FieldStore.store(field, METADATA, "test08.rf3", StoreVersion.V1)

    accessor = FieldStore.construct_field_accessor("test08.rf3")
    a_repr = repr(accessor)
    vx_count = accessor.get_voxel_count()
    true_vx_count = field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    b_repr = repr(accessor)
    assert a_repr == b_repr
    assert vx_count == true_vx_count
    assert accessor.get_field_type() == FieldType.CARTESIAN

    pickled_accessor = pickle.dumps(accessor)
    loaded_accessor = pickle.loads(pickled_accessor)

    assert loaded_accessor.get_voxel_count() == field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
    assert loaded_accessor.get_field_type() == accessor.get_field_type()
