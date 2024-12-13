#pragma once
#include "RadFiled3D/storage/Types.hpp"
#include <memory>

namespace RadFiled3D {
	namespace Storage {
		class MetadataSerializer {
		public:
			virtual void serializeMetadata(std::ostream& buffer, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const = 0;
		};

		namespace V1 {
			class MetadataSerializer : public RadFiled3D::Storage::MetadataSerializer {
			public:
				MetadataSerializer() = default;
				virtual void serializeMetadata(std::ostream& buffer, std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> metadata) const override;
			};
		};
	}
}