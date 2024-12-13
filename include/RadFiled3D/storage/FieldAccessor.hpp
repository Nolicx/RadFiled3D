#pragma once
#include <RadFiled3D/RadiationField.hpp>
#include "RadFiled3D/storage/Types.hpp"
#include <stdexcept>
#include <map>


namespace RadFiled3D {
	namespace Storage {
		class FieldAccessorBuilder;

		namespace AccessorTypes {
			typedef struct MemoryBlockDefinition {
				size_t offset;
				size_t size;

				MemoryBlockDefinition(size_t offset, size_t size)
					: offset(offset), size(size) {};

				MemoryBlockDefinition() {
					this->offset = 0;
					this->size = 0;
				}
			} MemoryBlockDefinition;

			typedef struct TypedMemoryBlockDefinition : public MemoryBlockDefinition {
			protected:
				std::vector<char> voxel_header_data;
			public:
				const char* get_voxel_header_data() const {
					return this->voxel_header_data.data();
				}

				Typing::DType dtype;
				size_t elements_per_voxel;

				TypedMemoryBlockDefinition(size_t offset, size_t size, Typing::DType dtype, size_t elements_per_voxel)
					: MemoryBlockDefinition(offset, size), dtype(dtype), elements_per_voxel(elements_per_voxel) {};

				TypedMemoryBlockDefinition() : MemoryBlockDefinition() {
					this->dtype = Typing::DType::Char;
					this->elements_per_voxel = 0;
				}

				void set_voxel_header_data(char* data, size_t size) {
					voxel_header_data.resize(size);
					memcpy(this->voxel_header_data.data(), data, size);
				}
			} TypedMemoryBlockDefinition;

			typedef struct ChannelStructure {
				MemoryBlockDefinition channel_block;
				std::map<std::string, TypedMemoryBlockDefinition> layers;

				ChannelStructure(MemoryBlockDefinition channel_block, const std::map<std::string, TypedMemoryBlockDefinition>& layers)
					: channel_block(channel_block), layers(layers) {};

				ChannelStructure() : channel_block(), layers() {};
			} ChannelStructure;
		}

		/** A class to access fields from a buffer
		* Descadent classes should implement the accessField method and should provide all needed methods to access the fields as a whole, by channel, by layer and by voxel.
		* The accessField method should return a shared pointer to the field, and the other methods should return shared pointers to the corresponding objects.
		* The FieldAccessor will be initialized from a template buffer in order to speed up loading of subsequent fields of the same structure e.g. files from the same dataset.
		* A new FieldAccessor requires adding FieldAccessorBuilder as a friend class, as FieldAccessors are created through it.
		*/
		class FieldAccessor {
			friend class RadFiled3D::Storage::FieldAccessorBuilder;
		private:
			const FieldType field_type;
		protected:
			size_t metadata_fileheader_size;
			size_t voxel_count = 0;

			/** Verify the buffer and set the read position to the beginning of the field.
			* @param buffer The buffer to verify
			* @throws RadiationFieldStoreException if the buffer is not valid
			*/
			void verifyBuffer(std::istream& buffer) const;

			/** Construct a field accessor with the given metadata size.
			* @param metadata_fileheader_size The size of the metadata in the file header
			* @param field_type The type of the field
			*/
			FieldAccessor(FieldType field_type);

			/** Initializes the accessor from a template buffer. */
			virtual void initialize(std::istream& buffer) = 0;
		public:
			/** Access a field from a buffer and return a shared pointer to it
			* @param buffer The buffer to access the field from
			* @return A shared pointer to the field
			*/
			virtual std::shared_ptr<IRadiationField> accessField(std::istream& buffer) const = 0;

			/** Get the version of the store that created a file
			* @param file The file to get the store version from
			* @return The store version of the file
			*/
			static StoreVersion getStoreVersion(std::istream& buffer);

			/** Quickly peeks at the mandatory metadata header of the radiation field from a buffer. Resets the file stream to the beginning
			* @param buffer The buffer to get the metadata from
			* @return The metadata header of the radiation field
			*/
			static FieldType peekFieldType(std::istream& file_stream);

			inline FieldType getFieldType() const {
				return this->field_type;
			}

			inline size_t getMetadataFileheaderOffset() const {
				return this->metadata_fileheader_size;
			}

			inline size_t getVoxelCount() const {
				return this->voxel_count;
			}

			/** Returns the offset from the beginning of a file to the start of the actual field data starting with the first channel block
			* @return The offset from the beginning of the file to the start of the field data
			*/
			virtual size_t getFieldDataOffset() const = 0;

			/** Accesses a voxel from a buffer and returns a pointer to it
			* @param buffer The buffer to access the voxel from
			* @param channel_name The name of the channel the voxel is in
			* @param layer_name The name of the layer the voxel is in
			* @param voxel_idx The index of the voxel in the layer
			* @return A pointer to the voxel
			*/
			virtual IVoxel* accessVoxelRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const = 0;

			/** Accesses a channel from a buffer and returns a shared pointer to it
			* @param buffer The buffer to access the channel from
			* @param channel_name The name of the channel to access
			* @return A shared pointer to the channel
			*/
			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxelFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const {
				IVoxel* voxel = this->accessVoxelRawFlat(buffer, channel_name, layer_name, voxel_idx);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};
		};

		class CartesianFieldAccessor : virtual public RadFiled3D::Storage::FieldAccessor {
		protected:
			CartesianFieldAccessor(FieldType field_type) : FieldAccessor(field_type) {};
		public:
			virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const = 0;
			virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const = 0;

			virtual std::shared_ptr<VoxelGridBuffer> accessChannel(std::istream& buffer, const std::string& channel_name) const = 0;
			virtual std::shared_ptr<VoxelGrid> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const = 0;

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxel(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const {
				IVoxel* voxel = this->accessVoxelRaw(buffer, channel_name, layer_name, voxel_idx);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxelByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const {
				auto voxel = this->accessVoxelRawByCoord(buffer, channel_name, layer_name, voxel_pos);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};
		};

		class PolarFieldAccessor : virtual public RadFiled3D::Storage::FieldAccessor {
		protected:
			PolarFieldAccessor(FieldType field_type) : FieldAccessor(field_type) {};

		public:
			virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const = 0;
			virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const = 0;

		public:
			virtual std::shared_ptr<PolarSegments> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const = 0;

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxel(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const {
				IVoxel* voxel = this->accessVoxelRaw(buffer, channel_name, layer_name, voxel_idx);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};

			template<typename dtype, typename VoxelT = ScalarVoxel<dtype>>
			std::shared_ptr<VoxelT> accessVoxelByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const {
				auto voxel = this->accessVoxelRawByCoord(buffer, channel_name, layer_name, voxel_pos);
				return std::shared_ptr<VoxelT>((VoxelT*)voxel);
			};
		};

		namespace V1 {
			class FileParser : virtual public RadFiled3D::Storage::FieldAccessor {
			protected:
				std::unique_ptr<RadFiled3D::Storage::V1::BinayFieldBlockHandler> serializer;

				FileParser(FieldType field_type) : FieldAccessor(field_type) {}
				

				/** Initialized the channels and layers structure of version 1 files.
				* Needs to be overridden by the different field shapes and this base method needs to be called in the overridden method right after parsing the channel header.
				*/
				virtual void initialize(std::istream& buffer) override;

				std::map<std::string, AccessorTypes::ChannelStructure> channels_layers_offsets;
			public:
				virtual IVoxel* accessVoxelRawFlat(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, size_t voxel_idx) const override;
			};


			class CartesianFieldAccessor : public RadFiled3D::Storage::CartesianFieldAccessor, public FileParser {
				friend class RadFiled3D::Storage::FieldAccessorBuilder;
			protected:
				glm::vec3 field_dimensions;
				glm::vec3 voxel_dimensions;
				std::unique_ptr<VoxelGrid> default_grid;

				CartesianFieldAccessor();
				virtual void initialize(std::istream& buffer) override;
			public:
				virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec3& voxel_idx) const override;
				virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec3& voxel_pos) const override;

				virtual std::shared_ptr<IRadiationField> accessField(std::istream& buffer) const override;
				virtual std::shared_ptr<VoxelGridBuffer> accessChannel(std::istream& buffer, const std::string& channel_name) const override;
				virtual std::shared_ptr<VoxelGrid> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const override;

				virtual size_t getFieldDataOffset() const override;
			};

			class PolarFieldAccessor : public RadFiled3D::Storage::PolarFieldAccessor, public FileParser {
				friend class RadFiled3D::Storage::FieldAccessorBuilder;
			protected:
				glm::uvec2 segments_counts;
				std::unique_ptr<PolarSegments> default_segments;

				PolarFieldAccessor();
				virtual void initialize(std::istream& buffer) override;
			public:
				virtual IVoxel* accessVoxelRaw(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::uvec2& voxel_idx) const override;
				virtual IVoxel* accessVoxelRawByCoord(std::istream& buffer, const std::string& channel_name, const std::string& layer_name, const glm::vec2& voxel_pos) const override;

				virtual std::shared_ptr<IRadiationField> accessField(std::istream& buffer) const override;
				virtual std::shared_ptr<PolarSegments> accessLayer(std::istream& buffer, const std::string& channel_name, const std::string& layer_name) const override;

				virtual size_t getFieldDataOffset() const override;
			};
		};

		class FieldAccessorBuilder {
		public:
			/** Construct a field accessor from a buffer
			* @param buffer The buffer to construct the field accessor from
			* @return The field accessor
			*/
			static std::shared_ptr<FieldAccessor> Construct(std::istream& buffer);
		};
	};
}
