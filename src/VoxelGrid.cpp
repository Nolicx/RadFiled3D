#include "RadFiled3D/VoxelGrid.hpp"


using namespace RadFiled3D;

VoxelGrid::VoxelGrid(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions, std::shared_ptr<VoxelLayer> layer)
	: voxel_dimensions(voxel_dimensions),
	  voxel_counts(glm::uvec3(field_dimensions / voxel_dimensions)),
	  layer(layer)
{
}

VoxelGridBuffer::VoxelGridBuffer(const glm::vec3& field_dimensions, const glm::vec3& voxel_dimensions)
	: voxel_grid(field_dimensions, voxel_dimensions),
	  VoxelBuffer(glm::uvec3(field_dimensions / voxel_dimensions).x * glm::uvec3(field_dimensions / voxel_dimensions).y * glm::uvec3(field_dimensions / voxel_dimensions).z)
{
}

VoxelBuffer* VoxelGridBuffer::copy() const
{
	glm::vec3 field_dimensions = glm::vec3(this->voxel_grid.get_voxel_counts()) * this->voxel_grid.get_voxel_dimensions();
	VoxelGridBuffer* copy = new VoxelGridBuffer(field_dimensions, this->get_voxel_dimensions());
	for (auto& layer : this->layers)
	{
		auto& layer_info = layer.second;
		IVoxel* initial_vx = (IVoxel*)layer_info.voxels;
		size_t bytes_per_voxel_databuffer = initial_vx->get_bytes();
		auto data = new char[this->voxel_count * bytes_per_voxel_databuffer];
		auto voxels = new char[this->voxel_count * layer_info.bytes_per_voxel];
		memcpy(data, layer_info.data, this->voxel_count * bytes_per_voxel_databuffer);
		memcpy(voxels, layer_info.voxels, this->voxel_count * layer_info.bytes_per_voxel);
		for (size_t i = 0; i < this->voxel_count; i++)
		{
			IVoxel* vx = (IVoxel*)(voxels + i * layer_info.bytes_per_voxel);
			vx->set_data((void*)(data + i * bytes_per_voxel_databuffer));
		}
		copy->layers[layer.first] = VoxelLayer(layer_info.bytes_per_voxel, layer_info.bytes_per_data_element, voxels, data, layer_info.unit, layer_info.statistical_error, this->voxel_count);
	}
	return copy;
}