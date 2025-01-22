#include "RadFiled3D/GridTracer.hpp"
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp> 
#include <iostream>
#include <set>


using namespace RadFiled3D;


std::vector<size_t> SamplingGridTracer::trace(const glm::vec3& p1, const glm::vec3& p2)
{
	std::vector<size_t> voxels;

	const glm::vec3 track_direction = glm::normalize(p2 - p1);
	const float track_length = glm::length(p2 - p1);

	const float track_step_length = std::min<float>(track_length, std::min<float>(std::min<float>(this->buffer.get_voxel_dimensions().x, this->buffer.get_voxel_dimensions().y), this->buffer.get_voxel_dimensions().z) / 2.f);
	const int track_possible_voxel_steps = static_cast<int>(track_length / track_step_length);

	for (int step_idx = 1; step_idx <= track_possible_voxel_steps; step_idx++) {
		const glm::vec3 pre_step_pos = p1 + track_direction * track_step_length * static_cast<float>(step_idx - 1);
		const glm::vec3 post_step_pos = p1 + track_direction * track_step_length * static_cast<float>(step_idx);

		if (post_step_pos.x < 0.f || post_step_pos.y < 0.f || post_step_pos.z < 0.f)
			continue;
		if (post_step_pos.x >= this->field_dimensions.x || post_step_pos.y >= this->field_dimensions.y || post_step_pos.z >= this->field_dimensions.z)
			continue;

		bool is_pre_step_outside = pre_step_pos.x < 0.f || pre_step_pos.y < 0.f || pre_step_pos.z < 0.f || pre_step_pos.x >= this->field_dimensions.x || pre_step_pos.y >= this->field_dimensions.y || pre_step_pos.z >= this->field_dimensions.z;

		size_t v1_idx = (is_pre_step_outside) ? 0 : this->buffer.get_voxel_idx_by_coord(pre_step_pos.x, pre_step_pos.y, pre_step_pos.z);
		size_t v2_idx = this->buffer.get_voxel_idx_by_coord(post_step_pos.x, post_step_pos.y, post_step_pos.z);

		if (v1_idx == v2_idx && !is_pre_step_outside)
			continue;

		voxels.push_back(v2_idx);
	}

	return voxels;
}

bool BresenhamGridTracer::isInside(const glm::ivec3& point) const
{
	const glm::uvec3 voxel_counts = buffer.get_voxel_counts();
	return (point.x >= 0 && point.x < static_cast<int>(voxel_counts.x) &&
		point.y >= 0 && point.y < static_cast<int>(voxel_counts.y) &&
		point.z >= 0 && point.z < static_cast<int>(voxel_counts.z));
}

std::vector<size_t> BresenhamGridTracer::trace(const glm::vec3& p1, const glm::vec3& p2)
{
	std::vector<size_t> voxels;

	bool excluded_first = false;

	glm::ivec3 ip1 = p1 / this->buffer.get_voxel_dimensions();
	glm::ivec3 ip2 = p2 / this->buffer.get_voxel_dimensions();

	glm::ivec3 d = glm::abs(ip2 - ip1);
	glm::ivec3 s = glm::ivec3(
		ip1.x < ip2.x ? 1 : -1,
		ip1.y < ip2.y ? 1 : -1,
		ip1.z < ip2.z ? 1 : -1
	);

	int err1, err2;
	if (d.x >= d.y && d.x >= d.z) {
		err1 = d.y - d.x / 2;
		err2 = d.z - d.x / 2;
		while (ip1.x != ip2.x) {
			if (this->isInside(ip1)) {
				if (excluded_first)
					voxels.push_back(this->buffer.get_voxel_idx(ip1.x, ip1.y, ip1.z));
				else
					excluded_first = true;
			}
			if (err1 >= 0) {
				ip1.y += s.y;
				err1 -= d.x;
			}
			if (err2 >= 0) {
				ip1.z += s.z;
				err2 -= d.x;
			}
			err1 += d.y;
			err2 += d.z;
			ip1.x += s.x;
		}
	}
	else if (d.y >= d.x && d.y >= d.z) {
		err1 = d.x - d.y / 2;
		err2 = d.z - d.y / 2;
		while (ip1.y != ip2.y) {
			if (this->isInside(ip1)) {
				if(excluded_first)
					voxels.push_back(this->buffer.get_voxel_idx(ip1.x, ip1.y, ip1.z));
				else
					excluded_first = true;
			}
			if (err1 >= 0) {
				ip1.x += s.x;
				err1 -= d.y;
			}
			if (err2 >= 0) {
				ip1.z += s.z;
				err2 -= d.y;
			}
			err1 += d.x;
			err2 += d.z;
			ip1.y += s.y;
		}
	}
	else {
		err1 = d.x - d.z / 2;
		err2 = d.y - d.z / 2;
		while (ip1.z != ip2.z) {
			if (this->isInside(ip1)) {
				if(excluded_first)
					voxels.push_back(this->buffer.get_voxel_idx(ip1.x, ip1.y, ip1.z));
				else
					excluded_first = true;
			}
			if (err1 >= 0) {
				ip1.x += s.x;
				err1 -= d.z;
			}
			if (err2 >= 0) {
				ip1.y += s.y;
				err2 -= d.z;
			}
			err1 += d.x;
			err2 += d.y;
			ip1.z += s.z;
		}
	}

	if (this->isInside(ip1)) {
		if (excluded_first)
			voxels.push_back(this->buffer.get_voxel_idx(ip1.x, ip1.y, ip1.z));
		else
			excluded_first = true;
	}

	return voxels;
}

std::vector<size_t> LinetracingGridTracer::trace(const glm::vec3& p1, const glm::vec3& p2)
{
	glm::vec3 line_start = p1;
	glm::vec3 line_end = p2;

	if (this->clipLine(line_start, line_end)) {
		bool clipped_incident = line_start != p1;
		const size_t start_voxel_idx = this->buffer.get_grid().get_voxel_idx_by_coord(line_start.x, line_start.y, line_start.z);
		auto voxels = this->lossyTracer.trace(line_start, line_end);
		if (clipped_incident) {
			if (this->buffer.get_voxel_count() > start_voxel_idx)
				voxels.push_back(start_voxel_idx);
		}
		std::set<size_t> voxels_to_test(voxels.begin(), voxels.end());
		for (size_t vx_idx : voxels) {
			const glm::uvec3 vx_indices = this->buffer.get_grid().get_voxel_indices(vx_idx);
			if (vx_indices.z + 1 < this->buffer.get_grid().get_voxel_counts().z)
				voxels_to_test.insert(this->buffer.get_grid().get_voxel_idx(vx_indices.x, vx_indices.y, vx_indices.z + 1));
			if (vx_indices.z > 0)
				voxels_to_test.insert(this->buffer.get_grid().get_voxel_idx(vx_indices.x, vx_indices.y, vx_indices.z - 1));
			if (vx_indices.y + 1 < this->buffer.get_grid().get_voxel_counts().y)
				voxels_to_test.insert(this->buffer.get_grid().get_voxel_idx(vx_indices.x, vx_indices.y + 1, vx_indices.z));
			if (vx_indices.y > 0)
				voxels_to_test.insert(this->buffer.get_grid().get_voxel_idx(vx_indices.x, vx_indices.y - 1, vx_indices.z));
			if (vx_indices.x + 1 < this->buffer.get_grid().get_voxel_counts().x)
				voxels_to_test.insert(this->buffer.get_grid().get_voxel_idx(vx_indices.x + 1, vx_indices.y, vx_indices.z));
			if (vx_indices.x > 0)
				voxels_to_test.insert(this->buffer.get_grid().get_voxel_idx(vx_indices.x - 1, vx_indices.y, vx_indices.z));
		}

		// perform line tracing on each cubic voxel in the list.
		std::vector<size_t> result;
		//const glm::vec3 voxel_half_dimension = this->buffer.get_voxel_dimensions() / 2.f;
		for (size_t vx_idx : voxels_to_test) {
			if (!clipped_incident && vx_idx == start_voxel_idx)
				continue;
			const glm::vec3 vx_pos = this->buffer.get_grid().get_voxel_coords(vx_idx);// -voxel_half_dimension;
			const glm::vec3 vx_pos_end = vx_pos + this->buffer.get_voxel_dimensions();

			if (this->intersectsAABB(line_start, line_end, vx_pos, vx_pos_end)) {
				result.push_back(vx_idx);
			}
		}
		return result;
	}
	return std::vector<size_t>();
}

bool LinetracingGridTracer::intersectsAABB(const glm::vec3& line_start, const glm::vec3& line_end, const glm::vec3& vx_pos, const glm::vec3& vx_pos_end) const
{
	glm::vec3 box_center = (vx_pos + vx_pos_end) * 0.5f;
	glm::vec3 box_half_size = (vx_pos_end - vx_pos) * 0.5f;
	glm::vec3 line_dir = line_end - line_start;
	glm::vec3 line_center = (line_start + line_end) * 0.5f;
	glm::vec3 line_half_size = glm::abs(line_dir) * 0.5f;
	glm::vec3 diff = line_center - box_center;

	glm::vec3 abs_line_dir = glm::abs(line_dir);

	if (glm::abs(diff.x) > (box_half_size.x + line_half_size.x)) return false;
	if (glm::abs(diff.y) > (box_half_size.y + line_half_size.y)) return false;
	if (glm::abs(diff.z) > (box_half_size.z + line_half_size.z)) return false;

	if (glm::abs(diff.y * line_dir.z - diff.z * line_dir.y) > (box_half_size.y * abs_line_dir.z + box_half_size.z * abs_line_dir.y)) return false;
	if (glm::abs(diff.z * line_dir.x - diff.x * line_dir.z) > (box_half_size.z * abs_line_dir.x + box_half_size.x * abs_line_dir.z)) return false;
	if (glm::abs(diff.x * line_dir.y - diff.y * line_dir.x) > (box_half_size.x * abs_line_dir.y + box_half_size.y * abs_line_dir.x)) return false;

	return true;
}

bool LinetracingGridTracer::clipLine(glm::vec3& start, glm::vec3& end) const
{
	float t0 = 0.0f, t1 = 1.0f;
	glm::vec3 d = end - start;

	const auto clip = [&](float p, float q) -> bool {
		if (p == 0 && q < 0) return false;
		float r = q / p;
		if (p < 0) {
			if (r > t1) return false;
			if (r > t0) t0 = r;
		}
		else if (p > 0) {
			if (r < t0) return false;
			if (r < t1) t1 = r;
		}
		return true;
	};

	if (clip(-d.x, start.x) && clip(d.x, this->gridDimensions.x - start.x) &&
		clip(-d.y, start.y) && clip(d.y, this->gridDimensions.y - start.y) &&
		clip(-d.z, start.z) && clip(d.z, this->gridDimensions.z - start.z)) {
		if (t1 < 1.0f)
			end = start + t1 * d;
		if (t0 > 0.0f)
			start = start + t0 * d;
		return true;
	}

	return false;
}

bool LinetracingGridTracer::isInside(const glm::vec3& point) const
{
	return  point.x >= 0 && point.x < this->gridDimensions.x &&
			point.y >= 0 && point.y < this->gridDimensions.y &&
			point.z >= 0 && point.z < this->gridDimensions.z;
}