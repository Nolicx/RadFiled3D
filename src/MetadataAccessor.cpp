#include "RadFiled3D/storage/MetadataAccessor.hpp"
#include "RadFiled3D/storage/Types.hpp"
#include <istream>

using namespace RadFiled3D;
using namespace RadFiled3D::Storage;
using namespace RadFiled3D::Storage::FiledTypes;

std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> RadFiled3D::Storage::V1::MetadataAccessor::accessMetadata(std::istream& buffer, bool quick_peek_only) const
{
	buffer.seekg(sizeof(VersionHeader), std::ios::beg);

	auto metadata = std::make_shared<RadFiled3D::Storage::V1::RadiationFieldMetadata>();
	metadata->deserialize(buffer, quick_peek_only);

	return metadata;
}

size_t RadFiled3D::Storage::V1::MetadataAccessor::get_metadata_size(std::istream& stream) const
{
	return this->meta_template.get_metadata_size(stream);
}