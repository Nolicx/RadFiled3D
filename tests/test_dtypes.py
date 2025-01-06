from RadFiled3D.RadFiled3D import vec2, vec3, vec4, uvec3, CartesianRadiationField
import numpy as np


def test_vec2():
    v1 = vec2(1.0, 2.0)
    assert v1.x == 1.0
    assert v1.y == 2.0
    assert v1 == vec2(1.0, 2.0)
    assert v1 != vec2(2.0, 1.0)
    assert v1 + vec2(1.0, 1.0) == vec2(2.0, 3)
    assert v1 - vec2(1.0, 1.0) == vec2(0.0, 1.0)
    assert v1 * 2.0 == vec2(2.0, 4)
    assert v1 * 2 == vec2(2.0, 4.0)
    assert v1 / 2.0 == vec2(0.5, 1.0)
    assert v1 * 2.0 == vec2(2.0, 4.0)
    assert v1 / 2.0 == 0.5 * v1

def test_vec3():
    v1 = vec3(1.0, 2.0, 3.0)
    assert v1.x == 1.0
    assert v1.y == 2.0
    assert v1.z == 3.0
    assert v1 == vec3(1.0, 2.0, 3.0)
    assert v1 != vec3(2.0, 1.0, 3.0)
    assert v1 + vec3(1.0, 1.0, 1.0) == vec3(2.0, 3, 4)
    assert v1 - vec3(1.0, 1.0, 1.0) == vec3(0.0, 1.0, 2.0)
    assert v1 * 2.0 == vec3(2.0, 4, 6)
    assert v1 / 2.0 == vec3(0.5, 1.0, 1.5)
    assert v1 * 2.0 == vec3(2.0, 4.0, 6.0)
    assert v1 / 2.0 == 0.5 * v1


def test_vec4():
    v1 = vec4(1.0, 2.0, 3.0, 4.0)
    assert v1.x == 1.0
    assert v1.y == 2.0
    assert v1.z == 3.0
    assert v1.w == 4.0
    assert v1 == vec4(1.0, 2.0, 3.0, 4.0)
    assert v1 != vec4(2.0, 1.0, 3.0, 4.0)
    assert v1 + vec4(1.0, 1.0, 1.0, 1.0) == vec4(2.0, 3, 4, 5)
    assert v1 - vec4(1.0, 1.0, 1.0, 1.0) == vec4(0.0, 1.0, 2.0, 3.0)
    assert v1 * 2.0 == vec4(2.0, 4, 6, 8)
    assert v1 / 2.0 == vec4(0.5, 1.0, 1.5, 2.0)
    assert v1 * 2.0 == vec4(2.0, 4.0, 6.0, 8.0)
    assert v1 / 2.0 == 0.5 * v1


def test_uvec3():
    v1 = uvec3(1, 2, 3)
    assert v1.x == 1
    assert v1.y == 2
    assert v1.z == 3
    assert v1 == uvec3(1, 2, 3)
    assert v1 != uvec3(2, 1, 3)
    assert v1 + uvec3(1, 1, 1) == uvec3(2, 3, 4)
    assert v1 - uvec3(1, 1, 1) == uvec3(0, 1, 2)
    assert v1 * 2 == uvec3(2, 4, 6)
    assert v1 / 2 == uvec3(0, 1, 1)
    assert v1 * 2 == uvec3(2, 4, 6)
    assert v1 / 2 == uvec3(0, 1, 1)

def test_radfield():
    field_dim = vec3(10, 10, 10)
    voxel_dim = vec3(1, 1, 1)
    field = CartesianRadiationField(field_dim, voxel_dim)
    field_voxels_count = field.get_voxel_counts()
    assert field_voxels_count.x == 10 and field_voxels_count.y == 10 and field_voxels_count.z == 10

    field_dim = vec3(0.768, 0.2, 0.768)
    voxel_dim = vec3(0.003, 0.004, 0.003)
    field = CartesianRadiationField(field_dim, voxel_dim)
    field_voxels_count = field.get_voxel_counts()
    assert field_voxels_count.x == 256 and field_voxels_count.y == 50 and field_voxels_count.z == 256

    field_dim = vec3(0.768, 0.204, 0.768)
    voxel_dim = vec3(0.003, 0.004, 0.003)
    field = CartesianRadiationField(field_dim, voxel_dim)
    field_voxels_count = field.get_voxel_counts()
    assert field_voxels_count.x == 256 and field_voxels_count.y == 51 and field_voxels_count.z == 256
