#include "RadFiled3D/PolarSegments.hpp"


using namespace RadFiled3D;

PolarSegments::PolarSegments(const glm::uvec2& segments_count, std::shared_ptr<VoxelLayer> layer)
	: segments_count(segments_count),
		layer(layer)
{
}

PolarSegmentsBuffer::PolarSegmentsBuffer(const glm::uvec2& segments_count)
	: segments(segments_count),
		VoxelBuffer(segments_count.x * segments_count.y)
{
}

VoxelBuffer* PolarSegmentsBuffer::copy() const
{
	PolarSegmentsBuffer* copy = new PolarSegmentsBuffer(this->segments.get_segments_count());
	for (auto& layer : this->layers)
	{
		auto& layer_info = layer.second;
		auto data = new char[this->voxel_count * layer_info.bytes_per_data_element];
		auto voxels = new char[this->voxel_count * layer_info.bytes_per_voxel];
		memcpy(data, layer_info.data, this->voxel_count * layer_info.bytes_per_data_element);
		memcpy(voxels, layer_info.voxels, this->voxel_count * layer_info.bytes_per_voxel);
		copy->layers[layer.first] = VoxelLayer(layer_info.bytes_per_voxel, layer_info.bytes_per_data_element, voxels, data, layer_info.unit, layer_info.statistical_error, this->voxel_count);
	}
	return copy;
}