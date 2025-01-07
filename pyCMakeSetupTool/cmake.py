from tempfile import TemporaryDirectory
import os
import subprocess
import multiprocessing
import shutil
import __main__
import tempfile
import re
import platform
import sysconfig
from typing import List, Tuple


class AlphaNumericSequence(tempfile._RandomNameSequence):
    characters = "abcdefghijklmnopqrstuvwxyz"


tempfile._name_sequence = AlphaNumericSequence()


class CMakeBuilder(object):
    """
    A Builder that takes a CMakeLists.txt with PyBind11 bindings to create a module.

    Args:
        build_target: Name of the build target to compile. Most likely will be the name of the python module.
        module_folder: The folder where the setup.py file is located.
        project_dir: The CMake Project Folder where the CMakeLists.txt is located.
        additional_binary_directories: A list of folder where relevant data is located which should be distributed.
        build_type: The type of build to perform out of: 'Release', 'Debug', 'RelWithDebInfo'.
        dependencies_file: A file which contains a line seperated list of additional binary directories.
        cmake_parameters_configure: A list of tuples with the cmake parameter name and value to be passed to the cmake configure command.
        partial_native_python_module_folder: The folder where the native python module code with additional python-files is located. This is used to copy the additional files to the build directory.
    """
    def __init__(self, build_target: str, stubs_dir: str, module_folder: str, project_dir: str = ".", additional_binary_directories: List[str] = [], build_type: str = "Release", dependencies_file: str = None, cmake_parameters_configure: List[Tuple[str, str]] = [], partial_native_python_module_folder: str = None) -> None:
        self.build_target = build_target
        self.stubs_dir = stubs_dir
        self.project_dir = project_dir
        self.build_type = build_type
        self.cmake_parameters_configure = cmake_parameters_configure if cmake_parameters_configure is not None else []
        self.dependencies_file = dependencies_file
        self.tmp_lib_dir = module_folder
        if not os.path.isabs(self.tmp_lib_dir):
            self.tmp_lib_dir = os.path.join(os.path.dirname(__main__.__file__), module_folder)
        self.additional_binary_directories = additional_binary_directories
        self.partial_native_python_module_folder = partial_native_python_module_folder
        if os.path.exists(self.tmp_lib_dir):
            shutil.rmtree(self.tmp_lib_dir)
        os.mkdir(self.tmp_lib_dir)
        if not os.path.isabs(self.project_dir):
            self.project_dir = os.path.join(os.path.dirname(__main__.__file__), self.project_dir)
        if not os.path.isabs(self.stubs_dir):
            self.stubs_dir = os.path.join(os.path.dirname(__main__.__file__), self.stubs_dir)
        if self.dependencies_file is not None and not os.path.isabs(self.dependencies_file):
            self.dependencies_file = os.path.join(os.path.dirname(__main__.__file__), self.dependencies_file)
        for i, path in enumerate(self.additional_binary_directories):
            if not os.path.isabs(path):
                self.additional_binary_directories[i] = os.path.join(os.path.dirname(__main__.__file__), path)
        if self.dependencies_file is not None:
            with open(self.dependencies_file, "r") as df:
                for ln in df.readlines():
                    ln = ln.replace("\n", "").replace("\r", "")
                    if len(ln) > 0:
                        path = ln if os.path.isabs(ln) else os.path.join(os.path.dirname(__main__.__file__), ln)
                        self.additional_binary_directories.append(path)

    def is_clang_installed(self) -> bool:
        has_clang = False
        try:
            subprocess.check_output(['clang', '--version'])
            has_clang = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        if not has_clang:
            try:
                subprocess.check_output(['clang-cl', '--version'])
                has_clang = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        return has_clang

    def write_init_py(self, to: str, include_dll_loading: bool = False, include_imports: bool = False):
        mod_imports = ""
        if include_imports:
            class_pattern = re.compile(r"(?:^|[\n\r])class\ +([^\(\:\ ]+)[\ ]*[:\(]{1}")
            for f in os.listdir(self.stubs_dir):
                if f.lower().endswith(".pyi"):
                    with open(os.path.join(self.stubs_dir, f), "r") as pyi:
                        cls_names = class_pattern.findall(pyi.read())
                        if len(cls_names) == 0:
                            continue
                        mod_name = "." + os.path.splitext(f)[0].replace("/", ".").replace("\\", ".")
                        mod_imports += f"from {mod_name} import "
                        for i, cn in enumerate(cls_names):
                            if i > 0:
                                mod_imports += ", "
                            mod_imports += cn
                    mod_imports += "\n"

        dll_loading = ""
        if include_dll_loading:
            dll_loading = f"""
lib_path = os.path.join(os.path.dirname(__file__), '{os.path.basename(self.get_binary_path())}')
lib = ctypes.CDLL(lib_path)
"""
        with open(to, "w") as f:
            f.write(f"""import ctypes
import os

# Module imports:
{mod_imports}

{dll_loading}
""")

    def run(self):
        with TemporaryDirectory() as dir:
            print("Start CMake configuration...")
            cmake_config_args = [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.join(dir, 'lib')}",
                #f"-DPYTHON_VERSION={sys.version_info[0]}.{sys.version_info[1]}"
            ]
            if os.name == 'nt' and self.is_clang_installed():
                cmake_config_args += ["-T", "ClangCL"]
            
            # Check if the system is 64-bit and add architecture, if is windows system
            if platform.machine().endswith('64') and os.name == 'nt':
                cmake_config_args += ['-A', 'x64']
            
            if len(self.cmake_parameters_configure) > 0:
                for param in self.cmake_parameters_configure:
                    cmake_config_args.append(f"-D{param[0]}={param[1]}")
            subprocess.check_call(['cmake'] + cmake_config_args + [self.project_dir], cwd=dir)
            print("Start CMake build...")
            subprocess.check_call([
                "cmake",
                "--build", ".",
                "--target", self.build_target,
                "--config", self.build_type,
                "-j", str(multiprocessing.cpu_count())
            ], cwd=dir)
            print("Build complete!")
            win_path = os.path.join(dir, 'lib', self.build_type)
            if os.path.exists(win_path):
                shutil.copytree(win_path, self.tmp_lib_dir, dirs_exist_ok=True)
            elif os.path.exists(os.path.join(dir, 'lib')):
                shutil.copytree(os.path.join(dir, 'lib'), self.tmp_lib_dir, dirs_exist_ok=True)
            else:
                raise Exception(f"Could not find build output directory! ({os.path.join(dir, 'lib')})")
        self.gather_artifacts()
    
    def gather_artifacts(self):
        shutil.copytree(self.stubs_dir, self.tmp_lib_dir, dirs_exist_ok=True)
        for bin_dir in self.additional_binary_directories:
            typed_dir = os.path.normpath(os.path.join(bin_dir, self.build_type))
            resolved_dir = typed_dir if os.path.exists(typed_dir) else bin_dir
            print("Copy artifact: " + resolved_dir)
            shutil.copytree(resolved_dir, self.tmp_lib_dir, dirs_exist_ok=True)
        if self.partial_native_python_module_folder is not None:
            shutil.copytree(self.partial_native_python_module_folder, self.tmp_lib_dir, dirs_exist_ok=True)
        self.write_init_py(os.path.join(self.tmp_lib_dir, "__init__.pyi"), True, False)
        self.write_init_py(os.path.join(self.tmp_lib_dir, "__init__.py"), False, False)

    def get_build_out_dir(self) -> str:
        return self.tmp_lib_dir

    def get_binary_path(self) -> str:
        # search for windows binary
        for f in os.listdir(self.tmp_lib_dir):
            if f.lower().endswith(".pyd"):
                return os.path.join(self.tmp_lib_dir, f)

        # search for linux binary
        for f in os.listdir(self.tmp_lib_dir):
            if f.lower().endswith(".so") and f.lower().startswith(self.build_target.lower()):
                return os.path.join(self.tmp_lib_dir, f)

        # if no binary was found, return None
        return None

    def get_build_name(self) -> str:
        return self.build_target


class CMakePreconfiguredBuilder(CMakeBuilder):
    """
    A CMake Builder that takes a preconfigured cmake directory for fast rebuild.
    This Builder should mostly be used for development builds.

    Args:
        build_target: Name of the build target to compile. Most likely will be the name of the python module.
        module_folder: The folder where the setup.py file is located.
        build_dir: Directory where the CMake binaries were already build and from where the build should be resumed.
        library_out_dir: Path where the resulting library will be built and from where the Extension should load the binary code.
        additional_binary_directories: A list of folder where relevant data is located which should be distributed.
        build_type: The type of build to perform out of: 'Release', 'Debug', 'RelWithDebInfo'.
        dependencies_file: A file which contains a line seperated list of additional binary directories.
    """
    def __init__(self, build_target: str, stubs_dir: str, module_folder: str, build_dir: str = ".", build_type: str = "Release", library_out_dir: str = "lib", additional_binary_directories: List[str] = [], dependencies_file: str = "Dependencies.conf") -> None:
        super().__init__(
            build_target=build_target,
            module_folder=module_folder,
            stubs_dir=stubs_dir,
            build_type=build_type,
            project_dir=".",
            additional_binary_directories=additional_binary_directories,
            dependencies_file=dependencies_file
        )
        self.build_dir = build_dir
        self.build_type = build_type
        if not os.path.isabs(self.build_dir):
            self.build_dir = os.path.join(os.getcwd(), self.build_dir)
        self.library_out_dir = library_out_dir
        if not os.path.isabs(self.library_out_dir):
            self.library_out_dir = os.path.join(os.getcwd(), self.library_out_dir)

    def run(self):
        subprocess.check_call([
            "cmake",
            "--build", ".",
            "--target", self.build_target,
            "--config", self.build_type,
            f"-j{multiprocessing.cpu_count()}"
        ], cwd=self.build_dir)
        print("Build complete!")
        shutil.copytree(os.path.join(self.library_out_dir, self.build_type), self.tmp_lib_dir, dirs_exist_ok=True)
        self.gather_artifacts()
