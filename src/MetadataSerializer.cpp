#include "RadFiled3D/storage/MetadataSerializer.hpp"
#include <ostream>

void RadFiled3D::Storage::V1::MetadataSerializer::serializeMetadata(std::ostream& buffer, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const
{
	if (metadata->get_version() != StoreVersion::V1)
		throw std::runtime_error("Metadata version mismatch");

	metadata->serialize(buffer);
}