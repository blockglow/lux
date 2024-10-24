use crate::gfx::{Surface as GfxSurface, SurfaceHandle};
use crate::prelude::*;

use super::include::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Surface(*const c_void);
impl Default for Surface {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl Surface {
    pub(crate) fn open(instance: Instance, gfx: &dyn GfxSurface) -> Result<Self, Error> {
        let SurfaceHandle::Linux { window, display } = gfx.handle() else {
            panic!("wrong platform for vulkan");
        };

        let surface_info = XlibSurfaceCreateInfo {
            s_type: StructureType::XlibSurfaceCreateInfo,
            p_next: ptr::null(),
            flags: 0,
            display,
            window,
        };

        let mut surface = Surface::default();
        VkResult::handle(unsafe {
            vkCreateXlibSurfaceKHR(instance, &surface_info, ptr::null(), &mut surface)
        })?;

        Ok(surface)
    }
}

#[repr(C)]
pub struct XlibSurfaceCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: u32,
    pub display: *const c_void,
    pub window: *const c_void,
}

#[repr(C)]
pub struct SurfaceCapabilitiesKHR {
    pub min_image_count: u32,
    pub max_image_count: u32,
    pub current_extent: Extent2D,
    pub min_image_extent: Extent2D,
    pub max_image_extent: Extent2D,
    pub max_image_array_layers: u32,
    pub supported_transforms: SurfaceTransformFlagsKHR,
    pub current_transform: SurfaceTransformFlagsKHR,
    pub supported_composite_alpha: CompositeAlphaFlagsKHR,
    pub supported_usage_flags: ImageUsageFlags,
}
