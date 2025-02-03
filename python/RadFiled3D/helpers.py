from RadFiled3D.RadFiled3D import vec3, RadiationFieldMetadataV1, RadiationFieldSoftwareMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSimulationMetadataV1


class MetadataHelper:
    @staticmethod
    def create_dummy_metadata() -> RadiationFieldMetadataV1:
        return RadiationFieldMetadataV1(
            RadiationFieldSimulationMetadataV1(
                0,
                "None",
                "None",
                RadiationFieldXRayTubeMetadataV1(
                    vec3(0, 0, 0),
                    vec3(0, 0, 0),
                    0,
                    "None"
                )
            ),
            RadiationFieldSoftwareMetadataV1(
                "None",
                "None",
                "None",
                "None",
                "None"
            )
        )
