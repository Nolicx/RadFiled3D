#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include <iostream>
#include "RadFiled3D/storage/RadiationFieldStore.hpp"
#include "RadFiled3D/storage/FieldAccessor.hpp"
#include <memory>
#include <vector>
#include <chrono>
#include <fstream>
#include "gtest/gtest.h"
#include <fstream>
#include <cstdio>
#include <thread>
#include <shared_mutex>
#ifdef _WIN32
#include <Windows.h>
#include "psapi.h"
#else
#include "sys/types.h"
#include "sys/sysinfo.h"
#endif

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;

namespace {
	class Storage : public ::testing::Test {
	protected:
		void TearDown() override {
			std::vector<std::string> files = { "test01.rf3" };
			for (auto& file : files) {
				if (std::ifstream(file).good())
					std::remove(file.c_str());
			}
		}
	};


	TEST(Storage, FieldAccessing) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("empty"));

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);

		std::shared_ptr<FieldAccessor> accessor = FieldStore::construct_accessor(file);

		file = std::ifstream("test01.rf3", std::ios::binary);
		std::shared_ptr<CartesianRadiationField> field2 = std::static_pointer_cast<CartesianRadiationField>(accessor->accessField(file));

		EXPECT_EQ(field->get_voxel_counts(), field2->get_voxel_counts());
		EXPECT_EQ(field->get_voxel_dimensions(), field2->get_voxel_dimensions());

		EXPECT_EQ(field->get_channels().size(), field2->get_channels().size());
		EXPECT_EQ(field->get_channels().at(0).first, field2->get_channels().at(0).first);

		EXPECT_EQ(field->get_channels().at(0).second->get_layers(), field2->get_channels().at(0).second->get_layers());

		auto channel1 = field->get_channel("test_channel");
		auto channel2 = field2->get_channel("test_channel");
		for (size_t i = 0; i < field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z; i++) {
			float val1 = channel1->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
			float val2 = channel2->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
			EXPECT_EQ(val1, val2);
		}
	}

	TEST(Storage, ChannelAccessing) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("empty"));

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);

		std::shared_ptr<FieldAccessor> accessor = FieldStore::construct_accessor(file);

		file = std::ifstream("test01.rf3", std::ios::binary);
		std::shared_ptr<VoxelGridBuffer> channel2 = std::dynamic_pointer_cast<CartesianFieldAccessor>(accessor)->accessChannel(file, "test_channel");


		EXPECT_EQ(channel->get_layers(), channel2->get_layers());
		EXPECT_EQ(channel->get_voxel_counts(), channel2->get_voxel_counts());

		for (size_t i = 0; i < field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z; i++) {
			float val1 = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
			float val2 = channel2->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
			EXPECT_EQ(val1, val2);
		}
	}

	TEST(Storage, LayerAccessing) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("empty"));

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);

		std::shared_ptr<FieldAccessor> accessor = FieldStore::construct_accessor(file);

		file = std::ifstream("test01.rf3", std::ios::binary);

		std::shared_ptr<VoxelGrid> layer = std::dynamic_pointer_cast<CartesianFieldAccessor>(accessor)->accessLayer(file, "test_channel", "doserate");

		auto channel1 = field->get_channel("test_channel");

		EXPECT_EQ(channel1->get_voxel_counts(), layer->get_voxel_counts());

		for (size_t i = 0; i < field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z; i++) {
			float val1 = channel1->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
			float val2 = layer->get_layer()->get_voxel_flat<ScalarVoxel<float>>(i).get_data();
			EXPECT_EQ(val1, val2);
		}
	}

	TEST(Storage, VoxelAccessing) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("empty"));

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);

		std::shared_ptr<FieldAccessor> accessor = FieldStore::construct_accessor(file);

		file = std::ifstream("test01.rf3", std::ios::binary);

		auto channel1 = field->get_channel("test_channel");

		for (size_t i = 0; i < field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z; i++) {
			auto voxel = std::dynamic_pointer_cast<CartesianFieldAccessor>(accessor)->accessVoxelFlat<float>(file, "test_channel", "doserate", i);
			float val1 = channel1->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
			float val2 = voxel->get_data();
			EXPECT_EQ(val1, val2);
		}
	}

	TEST(Storage, AccessingFromStringStream) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		field->add_channel("empty");

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);
		std::istringstream stream(std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()));
		std::shared_ptr<CartesianFieldAccessor> accessor = std::dynamic_pointer_cast<CartesianFieldAccessor>(FieldStore::construct_accessor(stream));

		file = std::ifstream("test01.rf3", std::ios::binary);
		stream = std::istringstream(std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()));

		auto channel2 = accessor->accessChannel(stream, "test_channel");
		EXPECT_EQ(channel->get_layers(), channel2->get_layers());
	}


	TEST(Storage, VoxelAccessingMemoryTest) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("empty"));

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);

		std::shared_ptr<FieldAccessor> accessor = FieldStore::construct_accessor(file);

		file = std::ifstream("test01.rf3", std::ios::binary);

		auto channel1 = field->get_channel("test_channel");

		// get memory consuption of this process under windows and linux
		unsigned long long totalVirtualUsed_begining = 0;
#ifdef _WIN32
		PROCESS_MEMORY_COUNTERS_EX pmc;
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_begining = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		struct sysinfo memInfo;
		sysinfo(&memInfo);

		totalVirtualUsed_begining = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_begining += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_begining *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		for (size_t y = 0; y < 100; y++) {
			for (size_t i = 0; i < field->get_voxel_counts().x * field->get_voxel_counts().y * field->get_voxel_counts().z; i++) {
				auto voxel = std::dynamic_pointer_cast<CartesianFieldAccessor>(accessor)->accessVoxelFlat<float>(file, "test_channel", "doserate", i);
				float val1 = channel1->get_voxel_flat<ScalarVoxel<float>>("doserate", i).get_data();
				float val2 = voxel->get_data();
				EXPECT_EQ(val1, val2);
			}
		}

		unsigned long long totalVirtualUsed_end = 0;
#ifdef _WIN32
		GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
		totalVirtualUsed_end = static_cast<unsigned long long>(pmc.PrivateUsage);
#else
		sysinfo(&memInfo);

		totalVirtualUsed_end = static_cast<unsigned long long>(memInfo.totalram - memInfo.freeram);
		totalVirtualUsed_end += static_cast<unsigned long long>(memInfo.totalswap - memInfo.freeswap);
		totalVirtualUsed_end *= static_cast<unsigned long long>(memInfo.mem_unit);
#endif

		std::cout << "Memory used in the begining: " << totalVirtualUsed_begining << std::endl;
		std::cout << "Memory used in the end: " << totalVirtualUsed_end << std::endl;
		unsigned long long memoryUsed = (totalVirtualUsed_end > totalVirtualUsed_begining) ? totalVirtualUsed_end - totalVirtualUsed_begining : 0;

		std::cout << "Memory used per field: " << memoryUsed / 1000 << std::endl;

		EXPECT_TRUE(memoryUsed / 1000 < sizeof(ScalarVoxel<float>) * 2 + 1);
	}

	TEST(Serialization, Self) {
		std::shared_ptr<CartesianRadiationField> field = std::make_shared<CartesianRadiationField>(glm::vec3(2.5f), glm::vec3(0.05f));
		std::shared_ptr<VoxelGridBuffer> channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("test_channel"));

		channel->add_layer<glm::vec3>("dirs", glm::vec3(0.f), "normalized direction");
		channel->add_layer<float>("doserate", 25.3f, "Gy/s");
		channel->add_custom_layer<HistogramVoxel>("spectra", HistogramVoxel(26, 10.f, nullptr), .123f, "");

		ScalarVoxel<float>& vx = channel->get_voxel_flat<ScalarVoxel<float>>("doserate", 20);
		vx = 10.f;

		channel = std::static_pointer_cast<VoxelGridBuffer>(field->add_channel("empty"));

		std::shared_ptr<RadFiled3D::Storage::V1::RadiationFieldMetadata> metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>(
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation(
				100,
				"geom",
				"FTFP_BERT",
				RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Simulation::XRayTube(
					glm::vec3(1.f, 0.f, 0.f),
					glm::vec3(0.f, 0.f, 0.f),
					100.f,
					"XRayTube"
				)
			),
			RadFiled3D::Storage::FiledTypes::V1::RadiationFieldMetadataHeader::Software(
				"test",
				"1.0",
				"repo",
				"commit"
			)
		);

		EXPECT_NO_THROW(FieldStore::store(field, metadata, "test01.rf3", StoreVersion::V1));

		std::ifstream file("test01.rf3", std::ios::binary);

		std::shared_ptr<FieldAccessor> accessor = FieldStore::construct_accessor(file);

		auto serialized = FieldAccessor::Serialize(accessor);

		std::shared_ptr<V1::CartesianFieldAccessor> accessor2 = std::dynamic_pointer_cast<V1::CartesianFieldAccessor>(FieldAccessor::Deserialize(serialized));

		EXPECT_EQ(accessor->getFieldType(), accessor2->getFieldType());
		EXPECT_EQ(accessor->getFieldDataOffset(), accessor2->getFieldDataOffset());
		EXPECT_EQ(accessor->getVoxelCount(), accessor2->getVoxelCount());
		{
			std::ifstream file("test01.rf3", std::ios::binary);
			EXPECT_NO_THROW(accessor2->accessField(file));
		}
		{
			std::ifstream file("test01.rf3", std::ios::binary);
			EXPECT_NO_THROW(accessor2->accessChannel(file, "test_channel"));
		}
		{
			std::ifstream file("test01.rf3", std::ios::binary);
			EXPECT_NO_THROW(accessor2->accessLayer(file, "test_channel", "doserate"));
		}
		{
			std::ifstream file("test01.rf3", std::ios::binary);
			auto vx = accessor2->accessVoxelFlat<float>(file, "test_channel", "doserate", 20);
			EXPECT_EQ(vx->get_data(), 10.f);
		}
	}
}