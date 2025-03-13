from RadFiled3D.RadFiled3D import vec3, RadiationFieldMetadataV1, RadiationFieldMetadataHeaderV1, RadiationFieldSoftwareMetadataV1, RadiationFieldXRayTubeMetadataV1, RadiationFieldSimulationMetadataV1
import __main__
import os


class Metadata(RadiationFieldMetadataV1):
    class Header(RadiationFieldMetadataHeaderV1):
        class Software(RadiationFieldSoftwareMetadataV1):
            pass

        class XRayTube(RadiationFieldXRayTubeMetadataV1):
            pass

        class Simulation(RadiationFieldSimulationMetadataV1):
            pass


    @staticmethod
    def default():
        """
        Returns a default metadata object for storing RadiationField files.
        Metadata contains the current main script name and default values in each field.
        Users may modify the metadata to describe the simulation.
        """

        sw_name = os.path.basename(__main__.__file__)
        return Metadata(
            simulation=Metadata.Header.Simulation(
                geometry="",
                primary_particle_count=0,
                physics_list="",
                tube=Metadata.Header.XRayTube(
                    vec3(0, 0, 0),
                    vec3(0, 0, 0),
                    0,
                    ""
                )
            ),
            software=Metadata.Header.Software(
                name=sw_name,
                version="",
                repository="",
                commit=""
            )
        )
