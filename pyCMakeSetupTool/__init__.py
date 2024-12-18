from setuptools import setup as setup_base, find_packages, Distribution
from setuptools.command.install import install
from setuptools.command.build_py import build_py
from typing import List, Tuple, Union, Type
import importlib.metadata
import sys
import os
import __main__
import site
from .cmake import CMakeBuilder


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


def write_manifest(additional_dirs: List[str]):
    m_path = os.path.join(os.path.dirname(__main__.__file__), "MANIFEST.in")
    if os.path.exists(m_path):
        os.remove(m_path)

    with open(m_path, "w+") as f:
        for ad in additional_dirs:
            p = os.path.normpath(os.path.relpath(ad, os.path.dirname(m_path)))
            f.write(f"graft {p}\n")


def setup(
    name: str,
    version: str,
    dependencies: list[str] = [],
    min_python: str = "3.8",
    author: str = None,
    author_email: str = None,
    license: str = None,
    cmake_builder: CMakeBuilder = None
    ) -> None:

    if os.environ.get("CI_COMMIT_TAG") is not None:
        version = os.environ.get("CI_COMMIT_TAG")

    ext_modules: List[Tuple[str, str]] = []
    additional_folders = []
    if cmake_builder is not None:
        cmake_builder.run()
        additional_folders.append(cmake_builder.get_build_out_dir())
        if cmake_builder.get_binary_path() is not None:
            ext_modules.append((name if cmake_builder.get_build_name() is None else cmake_builder.get_build_name(), cmake_builder.get_binary_path()))
    
    if len(additional_folders) > 0:
        write_manifest(additional_folders)

    distclass = BinaryDistribution if len(ext_modules) > 0 else None
    should_include_data_files = os.path.exists(os.path.join(os.path.dirname(__main__.__file__), "MANIFEST.in"))
    should_include_data_files = should_include_data_files or (len(additional_folders) > 0)

    packages = find_packages()
    package_data = {}
    if len(ext_modules) > 0:
        packages = list(set(packages + [m[0] for m in ext_modules]))
        for m in ext_modules:
            if m[0] not in package_data.keys():
                package_data[m[0]] = []
            package_data[m[0]].append(m[1])

    setup_base(
        name=name,
        version=version,
        packages=packages,
        package_data=package_data,
        include_package_data=should_include_data_files,
        distclass=distclass,
        install_requires=dependencies,
        python_requires=f">={min_python}",
        author=author,
        license=license,
        author_email=author_email
    )