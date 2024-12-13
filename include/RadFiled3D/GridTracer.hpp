#pragma once
#include <memory>
#include <vector>
#include "RadFiled3D/VoxelGrid.hpp"

namespace RadFiled3D {

	enum class GridTracerAlgorithm {
		SAMPLING = 0,
		BRESENHAM = 1,
		LINETRACING = 2
	};

	class GridTracer {
	protected:
		VoxelGridBuffer& buffer;
		const glm::vec3 field_dimensions;
	public:
		GridTracer(VoxelGridBuffer& buffer) 
			: buffer(buffer),
			  field_dimensions(glm::vec3(buffer.get_voxel_counts())* buffer.get_voxel_dimensions())
		{}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) = 0;
	};

	class SamplingGridTracer : public GridTracer {
	public:
		SamplingGridTracer(VoxelGridBuffer& buffer) : GridTracer(buffer) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;
	};

	class BresenhamGridTracer : public GridTracer {
	public:
		BresenhamGridTracer(VoxelGridBuffer& buffer) : GridTracer(buffer) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;

		bool isInside(const glm::ivec3& point) const;
	};

	class LinetracingGridTracer : public GridTracer {
	protected:
		const glm::vec3 gridDimensions;
	public:
		LinetracingGridTracer(VoxelGridBuffer& buffer)
			: GridTracer(buffer),
			  gridDimensions(glm::vec3(this->buffer.get_voxel_counts()) * this->buffer.get_voxel_dimensions()) {}

		virtual std::vector<size_t> trace(const glm::vec3& p1, const glm::vec3& p2) override;
		bool clipLine(glm::vec3& start, glm::vec3& end) const;
		bool isInside(const glm::vec3& point) const;
	};
}