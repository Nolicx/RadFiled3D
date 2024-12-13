#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/GridTracer.hpp"
#include <iostream>
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include <memory>
#include <vector>
#include <set>
#include <chrono>
#include <fstream>
#include "gtest/gtest.h"
#include <cstdio>

using namespace RadFiled3D;

namespace {
	TEST(Bresenham, TraceInside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		for (int i = 0; i < 9; i++)
			EXPECT_EQ(result[i], ((VoxelGridBuffer*)buffer.get())->get_voxel_idx_by_coord(0.1f * (i + 1), 0.1f * (i + 1), 0.1f * (i + 1)));
			

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.1f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 1);

		result = tracer.trace(glm::vec3(0.f), glm::vec3(0.2f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 2);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(1.f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 9);

		result = tracer.trace(glm::vec3(0.f, 0.5f, 0.5f), glm::vec3(0.5f, 0.5f, 0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);
	}

	TEST(Bresenham, TraceOutside) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-2.f), glm::vec3(-1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);

		result = tracer.trace(glm::vec3(2.f), glm::vec3(3.f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 0);
	}

	TEST(Bresenham, TraceAnywhere) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.1));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(-0.5f), glm::vec3(0.5f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(-0.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 5);

		result = tracer.trace(glm::vec3(0.5f), glm::vec3(2.5f));
		unique_result = std::set<size_t>(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 4);
	}

	TEST(Bresenham, TraceBigField) {
		CartesianRadiationField field(glm::vec3(1.f), glm::vec3(0.001));
		auto buffer = field.add_channel("test");

		BresenhamGridTracer tracer(*(VoxelGridBuffer*)buffer.get());

		auto result = tracer.trace(glm::vec3(0.f), glm::vec3(1.f));
		std::set<size_t> unique_result(result.begin(), result.end());
		EXPECT_EQ(result.size(), unique_result.size());
		EXPECT_EQ(result.size(), 998);
	}
}