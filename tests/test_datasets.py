try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False


def test_radfield3d_voxelwise_dataset():
    if TORCH_INSTALLED:
        from RadFiled3D.RadFiled3D import CartesianRadiationField, FieldStore, StoreVersion, FieldType, RadiationFieldMetadataV1, RadiationFieldSimulationMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSoftwareMetadataV1, vec3, uvec3, DType
        from RadFiled3D.pytorch.datasets.radfield3d import RadField3DVoxelwiseDataset, TrainingInputData
        import os
        import random
        import numpy as np

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
        ts = METADATA.add_dynamic_histogram_metadata("tube_spectrum", 150, 1.0)
        ts.get_histogram()[:] = np.arange(150, dtype=np.float32) * 0.01

        field = CartesianRadiationField(vec3(1, 1, 1), vec3(0.1, 0.1, 0.1))
        field.add_channel("scatter_field")
        field.add_channel("xray_beam")
        field.get_channel("scatter_field").add_layer("hits", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("scatter_field").add_histogram_layer("spectrum", 32, 0.1, "unit1")
        field.get_channel("xray_beam").add_layer("hits", "unit1", DType.FLOAT32)
        field.get_channel("xray_beam").add_layer("error", "unit1", DType.FLOAT32)
        field.get_channel("xray_beam").add_histogram_layer("spectrum", 32, 0.1, "unit1")

        os.makedirs("test_dataset", exist_ok=True)

        FieldStore.store(field, METADATA, "test_dataset/test01.rf3", StoreVersion.V1)
        FieldStore.store(field, METADATA, "test_dataset/test02.rf3", StoreVersion.V1)
        FieldStore.store(field, METADATA, "test_dataset/test03.rf3", StoreVersion.V1)

        dataset = RadField3DVoxelwiseDataset(file_paths=["test_dataset/test01.rf3", "test_dataset/test02.rf3", "test_dataset/test03.rf3"])
        ds_len = 3 * field.get_voxel_counts().x * field.get_voxel_counts().y * field.get_voxel_counts().z
        assert len(dataset) == ds_len, f"Dataset length does not match expected voxel count: {len(dataset)} != {ds_len}"

        test_in: TrainingInputData = dataset.__getitems__([random.randint(0, len(dataset)) for _ in range(100)])
        assert test_in.ground_truth.scatter_field.error.shape[0] == 100, "Ground truth error shape does not match expected batch size."
        assert test_in.ground_truth.scatter_field.fluence.shape[0] == 100, "Ground truth fluence shape does not match expected batch size."
        assert test_in.ground_truth.scatter_field.spectrum.shape[0] == 100, "Ground truth spectrum shape does not match expected batch size."
        assert test_in.ground_truth.xray_beam.error.shape[0] == 100, "X-ray beam error shape does not match expected batch size."
        assert test_in.ground_truth.xray_beam.fluence.shape[0] == 100, "X-ray beam fluence shape does not match expected batch size."
        assert test_in.ground_truth.xray_beam.spectrum.shape[0] == 100, "X-ray beam spectrum shape does not match expected batch size."
        assert test_in.input.direction.shape[0] == 100, "Input direction shape does not match expected batch size."
        assert test_in.input.position.shape[0] == 100, "Input position shape does not match expected batch size."
        assert test_in.input.spectrum.shape[0] == 100, "Input tube spectrum shape does not match expected batch size."
