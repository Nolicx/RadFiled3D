#include "RadFiled3D/storage/FieldAccessor.hpp"
#include "RadFiled3D/RadiationField.hpp"
#include "RadFiled3D/Voxel.hpp"
#include "RadFiled3D/VoxelGrid.hpp"
#include "RadFiled3D/PolarSegments.hpp"
#include "RadFiled3D/storage/FieldSerializer.hpp"
#include "RadFiled3D/storage/MetadataAccessor.hpp"
#include <istream>
#include <memory>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;
using namespace RadFiled3D::Storage::FiledTypes;


StoreVersion RadFiled3D::Storage::FieldAccessor::getStoreVersion(std::istream& buffer)
{
	VersionHeader version;
	buffer.seekg(0, std::ios::beg);
	buffer.read((char*)&version, sizeof(VersionHeader));

	if (strcmp(version.version, "1.0") == 0)
		return StoreVersion::V1;

	throw RadiationFieldStoreException(std::string("Unsupported file version: ") + std::string(version.version));
}

void RadFiled3D::Storage::FieldAccessor::verifyBuffer(std::istream& buffer) const
{
	/*buffer.seekg(this->metadata_fieldheader_verification_position.first, std::ios::beg);
	char verifyByte = 0;
	buffer.read(&verifyByte, 1);
	if (verifyByte != this->metadata_fieldheader_verification_position.second)
		throw RadiationFieldStoreException("Field header verification failed");
	*/

	buffer.seekg(this->metadata_fileheader_size, std::ios::beg);
}

RadFiled3D::Storage::FieldAccessor::FieldAccessor(FieldType field_type)
	: field_type(field_type), metadata_fileheader_size(0)
{
}

void RadFiled3D::Storage::V1::FileParser::initialize(std::istream& buffer)
{
	if (this->voxel_count == 0) {
		throw RadiationFieldStoreException("Invalid voxel count");
	}

	this->channels_layers_offsets.clear();
	buffer.seekg(this->getFieldDataOffset(), std::ios::beg);

	this->serializer = std::make_unique<BinayFieldBlockHandler>();

	size_t channel_pos = 0;
	const size_t start_pos = buffer.tellg();
	buffer.seekg(0, std::ios::end);
	const size_t max_bytes = buffer.tellg();
	buffer.seekg(start_pos);
	while (!buffer.eof() && channel_pos < max_bytes) {
		FiledTypes::V1::ChannelHeader channel_header;
		buffer.read((char*)&channel_header, sizeof(FiledTypes::V1::ChannelHeader));

		if (std::string(channel_header.name).empty()) {
			channel_pos += sizeof(FiledTypes::V1::ChannelHeader);
			buffer.seekg(this->getFieldDataOffset() + channel_pos, std::ios::beg);
			continue;
		}

		AccessorTypes::MemoryBlockDefinition channel_block(channel_pos, channel_header.channel_bytes);
		std::map<std::string, AccessorTypes::TypedMemoryBlockDefinition> layers_blocks;

		size_t layer_pos = 0;
		while (layer_pos + sizeof(FiledTypes::V1::VoxelGridLayerHeader) < channel_header.channel_bytes) {
			FiledTypes::V1::VoxelGridLayerHeader layer_header;
			buffer.seekg(this->getFieldDataOffset() + sizeof(FiledTypes::V1::ChannelHeader) + layer_pos + channel_pos, std::ios::beg);
			buffer.read((char*)&layer_header, sizeof(FiledTypes::V1::VoxelGridLayerHeader));

			const size_t layer_size = sizeof(FiledTypes::V1::VoxelGridLayerHeader) + this->voxel_count * layer_header.bytes_per_element + layer_header.header_block_size;
			Typing::DType dtype = Typing::Helper::get_dtype(std::string(layer_header.dtype));
			size_t elements_per_voxel = layer_header.bytes_per_element / Typing::Helper::get_bytes_of_dtype(dtype);
			layers_blocks[layer_header.name] = AccessorTypes::TypedMemoryBlockDefinition(layer_pos, layer_size, dtype, elements_per_voxel);

			if (layer_header.header_block_size > 0) {
				char* header_data = new char[layer_header.header_block_size];
				buffer.read(header_data, layer_header.header_block_size);
				layers_blocks[layer_header.name].set_voxel_header_data(header_data, layer_header.header_block_size);
				delete[] header_data;
			}

			layer_pos += layer_size;
		}
		this->channels_layers_offsets[channel_header.name] = AccessorTypes::ChannelStructure(channel_block, layers_blocks);
		channel_pos += channel_header.channel_bytes + sizeof(FiledTypes::V1::ChannelHeader);
		buffer.seekg(this->getFieldDataOffset() + channel_pos, std::ios::beg);
	}
}

IVoxel* RadFiled3D::Storage::V1::FileParser::accessVoxelRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	const size_t element_size = Typing::Helper::get_bytes_of_dtype(layer_block.dtype);

	if (voxel_idx >= this->voxel_count)
		throw RadiationFieldStoreException("Voxel index out of bounds");

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + sizeof(FiledTypes::V1::VoxelGridLayerHeader) + sizeof(FiledTypes::V1::ChannelHeader) + voxel_idx * layer_block.elements_per_voxel * element_size, std::ios::beg);
	char* data_buffer = new char[layer_block.elements_per_voxel * element_size];
	buffer.read(data_buffer, layer_block.elements_per_voxel * element_size);

	IVoxel* voxel = nullptr;
	switch (layer_block.dtype) {
	case Typing::DType::Float:
		voxel = new OwningScalarVoxel<float>((float*)data_buffer);
		break;
	case Typing::DType::Double:
		voxel = new OwningScalarVoxel<double>((double*)data_buffer);
		break;
	case Typing::DType::Int:
		voxel = new OwningScalarVoxel<int>((int*)data_buffer);
		break;
	case Typing::DType::UInt32:
		voxel = new OwningScalarVoxel<unsigned long>((unsigned long*)data_buffer);
		break;
	case Typing::DType::UInt64:
		voxel = new OwningScalarVoxel<unsigned long long>((unsigned long long*)data_buffer);
		break;
	case Typing::DType::Char:
		voxel = new OwningScalarVoxel<char>((char*)data_buffer);
		break;
	case Typing::DType::Vec2:
		voxel = new OwningScalarVoxel<glm::vec2>((glm::vec2*)data_buffer);
		break;
	case Typing::DType::Vec3:
		voxel = new OwningScalarVoxel<glm::vec3>((glm::vec3*)data_buffer);
		break;
	case Typing::DType::Vec4:
		voxel = new OwningScalarVoxel<glm::vec4>((glm::vec4*)data_buffer);
		break;
	}

	if (voxel == nullptr && layer_block.dtype == Typing::DType::Hist) {
		OwningHistogramVoxel* vx = new OwningHistogramVoxel();
		if (layer_block.get_voxel_header_data() != nullptr) {
			vx->init_from_header(layer_block.get_voxel_header_data());
		}
		vx->set_data(data_buffer);
		voxel = vx;
	}

	delete[] data_buffer;

	if (voxel == nullptr)
		throw RadiationFieldStoreException("Unsupported data type");
	return voxel;
}

size_t RadFiled3D::Storage::V1::CartesianFieldAccessor::getFieldDataOffset() const
{
	return this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader) + sizeof(FiledTypes::V1::CartesianHeader);
}

size_t RadFiled3D::Storage::V1::PolarFieldAccessor::getFieldDataOffset() const
{
	return this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader) + sizeof(FiledTypes::V1::PolarHeader);
}

void RadFiled3D::Storage::V1::CartesianFieldAccessor::initialize(std::istream& buffer)
{
	this->metadata_fileheader_size = RadFiled3D::Storage::V1::MetadataAccessor().get_metadata_size(buffer) + sizeof(Storage::FiledTypes::VersionHeader);

	buffer.seekg(this->getMetadataFileheaderOffset() + sizeof(FiledTypes::V1::RadiationFieldHeader), std::ios::beg);
	FiledTypes::V1::CartesianHeader ch;
	buffer.read((char*)&ch, sizeof(FiledTypes::V1::CartesianHeader));

	this->field_dimensions = glm::vec3(ch.voxel_counts) * ch.voxel_dimensions;
	this->voxel_dimensions = ch.voxel_dimensions;
	this->voxel_count = ch.voxel_counts.x * ch.voxel_counts.y * ch.voxel_counts.z;
	this->default_grid = std::make_unique<VoxelGrid>(this->field_dimensions, ch.voxel_dimensions);

	FileParser::initialize(buffer);
}

RadFiled3D::Storage::V1::CartesianFieldAccessor::CartesianFieldAccessor()
	: RadFiled3D::Storage::FieldAccessor(FieldType::Cartesian),
	  RadFiled3D::Storage::CartesianFieldAccessor(FieldType::Cartesian),
	  RadFiled3D::Storage::V1::FileParser(FieldType::Cartesian),
	  field_dimensions(glm::vec3(0)),
	  voxel_dimensions(glm::vec3(0.f))
{
}

std::shared_ptr<IRadiationField> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessField(std::istream& buffer) const
{
	size_t metadata_size = this->getMetadataFileheaderOffset();
	buffer.seekg(metadata_size, std::ios::beg);
	return this->serializer->deserializeField(buffer);
}

std::shared_ptr<VoxelGridBuffer> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessChannel(std::istream& buffer, const std::string& channel_name) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto& channel_block = channel_block_itr->second.channel_block;

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);

	auto grid_buffer = std::make_shared<VoxelGridBuffer>(this->field_dimensions, this->voxel_dimensions);
	char* data_buffer = new char[channel_block.size];
	buffer.read(data_buffer, channel_block.size);
	this->serializer->deserializeChannel(grid_buffer, data_buffer, channel_block.size);
	delete[] data_buffer;

	return grid_buffer;
}

std::shared_ptr<VoxelGrid> RadFiled3D::Storage::V1::CartesianFieldAccessor::accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);

	char* data_buffer = new char[layer_block.size];
	buffer.read(data_buffer, layer_block.size);
	VoxelLayer* layer = this->serializer->deserializeLayer(data_buffer, layer_block.size);
	delete[] data_buffer;

	return std::make_shared<VoxelGrid>(this->field_dimensions, this->voxel_dimensions, std::shared_ptr<VoxelLayer>(layer));
}

IVoxel* RadFiled3D::Storage::V1::CartesianFieldAccessor::accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const
{
	const size_t idx = this->default_grid->get_voxel_idx(voxel_idx.x, voxel_idx.y, voxel_idx.z);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

IVoxel* RadFiled3D::Storage::V1::CartesianFieldAccessor::accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const
{
	const size_t idx = this->default_grid->get_voxel_idx_by_coord(voxel_pos.x, voxel_pos.y, voxel_pos.z);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

void RadFiled3D::Storage::V1::PolarFieldAccessor::initialize(std::istream& buffer)
{
	this->metadata_fileheader_size = RadFiled3D::Storage::V1::MetadataAccessor().get_metadata_size(buffer) + sizeof(Storage::FiledTypes::VersionHeader);
	buffer.seekg(sizeof(VersionHeader) + sizeof(FiledTypes::V1::RadiationFieldHeader), std::ios::beg);
	FiledTypes::V1::PolarHeader ph;
	buffer.read((char*)&ph, sizeof(FiledTypes::V1::PolarHeader));

	this->segments_counts = ph.segments_counts;
	this->voxel_count = ph.segments_counts.x * ph.segments_counts.y;
	this->default_segments = std::make_unique<PolarSegments>(ph.segments_counts);

	FileParser::initialize(buffer);
}

RadFiled3D::Storage::V1::PolarFieldAccessor::PolarFieldAccessor()
	: RadFiled3D::Storage::FieldAccessor(FieldType::Polar),
	  RadFiled3D::Storage::PolarFieldAccessor(FieldType::Polar),
	  RadFiled3D::Storage::V1::FileParser(FieldType::Polar),
	  segments_counts(glm::uvec2(0))
{
}

IVoxel* RadFiled3D::Storage::V1::PolarFieldAccessor::accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const
{
	size_t idx = this->default_segments->get_segment_idx(voxel_idx.x, voxel_idx.y);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

IVoxel* RadFiled3D::Storage::V1::PolarFieldAccessor::accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const
{
	size_t idx = this->default_segments->get_segment_idx_by_coord(voxel_pos.x, voxel_pos.y);
	return this->accessVoxelRawFlat(buffer, channel_name, layer_name, idx);
}

std::shared_ptr<IRadiationField> RadFiled3D::Storage::V1::PolarFieldAccessor::accessField(std::istream& buffer) const
{
	buffer.seekg(this->getMetadataFileheaderOffset(), std::ios::beg);
	return this->serializer->deserializeField(buffer);
}

std::shared_ptr<PolarSegments> RadFiled3D::Storage::V1::PolarFieldAccessor::accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const
{
	auto channel_block_itr = this->channels_layers_offsets.find(channel_name);
	if (channel_block_itr == this->channels_layers_offsets.end())
		throw RadiationFieldStoreException("Channel not found");

	auto layer_block_itr = channel_block_itr->second.layers.find(layer_name);
	if (layer_block_itr == channel_block_itr->second.layers.end())
		throw RadiationFieldStoreException("Layer not found");

	auto& channel_block = channel_block_itr->second.channel_block;
	auto& layer_block = layer_block_itr->second;

	buffer.seekg(this->getFieldDataOffset() + channel_block.offset + layer_block.offset + sizeof(FiledTypes::V1::ChannelHeader), std::ios::beg);

	char* data_buffer = new char[layer_block.size];
	buffer.read(data_buffer, layer_block.size);
	PolarSegments* layer = (PolarSegments*)this->serializer->deserializeLayer(data_buffer, layer_block.size);
	delete[] data_buffer;

	return std::shared_ptr<PolarSegments>(layer);
}

std::shared_ptr<FieldAccessor> FieldAccessorBuilder::Construct(std::istream& buffer)
{
	StoreVersion version = FieldAccessor::getStoreVersion(buffer);

	std::shared_ptr<FieldAccessor> accessor;

	switch (version)
	{
	case StoreVersion::V1:
		buffer.seekg(RadFiled3D::Storage::V1::MetadataAccessor().get_metadata_size(buffer) + sizeof(Storage::FiledTypes::VersionHeader), std::ios::beg);
		switch (RadFiled3D::Storage::V1::BinayFieldBlockHandler().getFieldType(buffer)) {
		case FieldType::Cartesian:
			accessor = std::static_pointer_cast<FieldAccessor>(std::shared_ptr<Storage::V1::CartesianFieldAccessor>(new Storage::V1::CartesianFieldAccessor()));
			break;
		case FieldType::Polar:
			accessor = std::static_pointer_cast<FieldAccessor>(std::shared_ptr<Storage::V1::PolarFieldAccessor>(new Storage::V1::PolarFieldAccessor()));
			break;
		default:
			throw RadiationFieldStoreException("Unsupported field type");
		}
		break;
	default:
		throw RadiationFieldStoreException("Unsupported file version");
	}

	accessor->initialize(buffer);

	return accessor;
}
