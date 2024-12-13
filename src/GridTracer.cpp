#include "RadFiled3D/GridTracer.hpp"
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/component_wise.hpp> 
#include <iostream>


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
	throw std::runtime_error("Not implemented");
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