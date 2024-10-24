use crate::prelude::*;

use super::include::*;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Image(*const c_void);
impl Default for Image {
    fn default() -> Self {
        Image(ptr::null())
    }
}

// Flag bits for ImageAspectFlags
#[repr(u32)]
pub enum ImageAspectFlagBits {
    Color = 0x00000001,
    Depth = 0x00000002,
    Stencil = 0x00000004,
    Metadata = 0x00000008,
    // Add other variants as per Vulkan spec
}

#[repr(C)]
pub enum ImageLayout {
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    Present = 1000001002,
    // ... other variant1_000_001_002s ...
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ImageView(*const c_void);
impl Default for ImageView {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl ImageView {
    pub(crate) fn create(
        device: Device,
        image: Image,
        view_type: ImageViewType,
        format: Format,
    ) -> Result<Self, Error> {
        let mut image_view = ImageView::default();
        let info = ImageViewCreateInfo {
            s_type: StructureType::ImageViewCreateInfo,
            p_next: ptr::null(),
            flags: 0,
            image,
            view_type,
            format,
            components: ComponentMapping {
                r: ComponentSwizzle::Identity,
                g: ComponentSwizzle::Identity,
                b: ComponentSwizzle::Identity,
                a: ComponentSwizzle::Identity,
            },
            subresource_range: ImageSubresourceRange {
                aspect_mask: ImageAspectFlagBits::Color as u32,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        VkResult::handle(unsafe {
            vkCreateImageView(device, &info, ptr::null(), &mut image_view)
        })?;
        Ok(image_view)
    }
}

#[repr(C)]
pub struct ImageViewCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: ImageViewCreateFlags,
    pub image: Image,
    pub view_type: ImageViewType,
    pub format: Format,
    pub components: ComponentMapping,
    pub subresource_range: ImageSubresourceRange,
}

pub type ImageViewCreateFlags = u32;

#[repr(C)]
pub enum ImageViewType {
    Type1d = 0,
    Type2d = 1,
    Type3d = 2,
    Cube = 3,
    Type1dArray = 4,
    Type2dArray = 5,
    CubeArray = 6,
}

#[repr(C)]
pub struct ComponentMapping {
    pub r: ComponentSwizzle,
    pub g: ComponentSwizzle,
    pub b: ComponentSwizzle,
    pub a: ComponentSwizzle,
}

#[repr(C)]
pub enum ComponentSwizzle {
    Identity = 0,
    Zero = 1,
    One = 2,
    R = 3,
    G = 4,
    B = 5,
    A = 6,
}

#[repr(C)]
pub struct ImageSubresourceRange {
    pub aspect_mask: ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

pub type ImageAspectFlags = u32;

pub type ImageUsageFlags = u32;

#[repr(u32)]
pub enum ImageUsageFlagBits {
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    Sampled = 0x00000004,
    Storage = 0x00000008,
    ColorAttachment = 0x00000010,
    DepthStencilAttachment = 0x00000020,
    TransientAttachment = 0x00000040,
    InputAttachment = 0x00000080,
}
