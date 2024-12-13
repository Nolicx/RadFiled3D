#pragma once
#include "RadFiled3D/storage/Types.hpp"
#include <memory>

namespace RadFiled3D {
	namespace Storage {
		class MetadataAccessor {
		public:
			virtual std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> accessMetadata(std::istream& buffer, bool quick_peek_only = false) const = 0;
			virtual size_t get_metadata_size(std::istream& stream) const = 0;
		};

		namespace V1 {
			class MetadataAccessor : public RadFiled3D::Storage::MetadataAccessor {
			protected:
				RadFiled3D::Storage::V1::RadiationFieldMetadata meta_template;
			public:
				virtual std::shared_ptr<RadFiled3D::Storage::RadiationFieldMetadata> accessMetadata(std::istream& buffer, bool quick_peek_only = false) const override;
				virtual size_t get_metadata_size(std::istream& stream) const override;
			};
		};
	}
}