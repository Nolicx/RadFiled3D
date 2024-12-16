from fb63SetupTools import setup
from fb63SetupTools.builders import CMakeBuilder, CMakePreconfiguredBuilder
import sys
import platform
import os


IS_DEVELOPMENT_BUILD = False
BUILD_PATH = ""


if "--development" in sys.argv:
    v_idx = sys.argv.index("--development")
    sys.argv.pop(v_idx)
    BUILD_PATH = sys.argv[v_idx]
    sys.argv.pop(v_idx)
    IS_DEVELOPMENT_BUILD = True


major_minor_version = '.'.join(platform.python_version().split('.')[:2])

setup(
    default_name="RadFiled3D",
    default_version="1.0.0",
    dependencies=[
        "numpy>=2.0.0",
        "rich"
    ],
    custom_builders=[
        CMakePreconfiguredBuilder(
            build_target="RadFiled3D",
            stubs_dir=os.path.join(os.path.dirname(__file__), "python/stubs"),
            build_dir=BUILD_PATH,
            library_out_dir=BUILD_PATH,
            module_folder=os.path.join(os.path.dirname(__file__), "RadFiled3D"),
            dependencies_file=None
        ) if IS_DEVELOPMENT_BUILD else CMakeBuilder(
            build_target="RadFiled3D",
            stubs_dir=os.path.join(os.path.dirname(__file__), "python/stubs"),
            module_folder=os.path.join(os.path.dirname(__file__), "RadFiled3D"),
            project_dir=os.path.dirname(__file__),
            dependencies_file=None,
            partial_native_python_module_folder=os.path.join(os.path.dirname(__file__), "python/RadFiled3D"),
            cmake_parameters_configure=[
                ("BUILD_TESTS", "OFF"),
                ("BUILD_EXAMPLES", "OFF"),
                ("PYTHON_VERSION", major_minor_version),
                ("Python_ROOT_DIR", sys.exec_prefix)
            ]
        )
    ]
)
