use crate::prelude::*;

use super::include::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Swapchain(*const c_void);
impl Default for Swapchain {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl Swapchain {
    pub(crate) fn next_image(self, device: Device, semaphore: Semaphore) -> Result<u32, Error> {
        let mut index = 0;
        VkResult::handle(unsafe {
            vkAcquireNextImageKHR(
                device,
                self,
                u64::MAX,
                semaphore,
                Fence::default(),
                &mut index,
            )
        })?;
        Ok(index)
    }
    pub(crate) fn new(
        device: Device,
        queue_family_index: QueueFamilyIndex,
        surface: Surface,
        old_swapchain: Option<Swapchain>,
        (width, height): (u32, u32),
    ) -> Result<(Swapchain, Vec<Image>), Error> {
        let swapchain_info = SwapchainCreateInfo {
            s_type: StructureType::SwapchainCreateInfo,
            p_next: ptr::null(),
            flags: 0,
            surface,
            min_image_count: 3,
            image_format: Format::Bgra8Unorm,
            image_color_space: ColorSpace::SrgbNonlinear,
            image_extent: Extent2D { width, height },
            image_array_layers: 1,
            image_usage: ImageUsageFlagBits::ColorAttachment as u32,
            image_sharing_mode: SharingMode::Exclusive,
            queue_family_index_count: 1,
            p_queue_family_indices: &queue_family_index,
            pre_transform: SurfaceTransformFlagBits::IdentityBit,
            composite_alpha: CompositeAlphaFlagBits::OpaqueBit,
            present_mode: PresentMode::Immediate,
            clipped: 0,
            old_swapchain,
        };

        let mut swapchain = Swapchain::default();
        VkResult::handle(dbg!(unsafe {
            vkCreateSwapchainKHR(device, &swapchain_info, ptr::null(), &mut swapchain)
        }))?;

        let mut count = 8;
        let mut images = vec![Image::default(); count as usize];

        unsafe { vkGetSwapchainImagesKHR(device, swapchain, &mut count, images.as_mut_ptr()) }

        images.truncate(count as usize);
        dbg!(&images);
        Ok((swapchain, images))
    }
}
#[repr(C)]
pub struct SwapchainCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: SwapchainCreateFlags,
    pub surface: Surface,
    pub min_image_count: u32,
    pub image_format: Format,
    pub image_color_space: ColorSpace,
    pub image_extent: Extent2D,
    pub image_array_layers: u32,
    pub image_usage: ImageUsageFlags,
    pub image_sharing_mode: SharingMode,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const QueueFamilyIndex,
    pub pre_transform: SurfaceTransformFlagBits,
    pub composite_alpha: CompositeAlphaFlagBits,
    pub present_mode: PresentMode,
    pub clipped: Bool32,
    pub old_swapchain: Option<Swapchain>,
}

pub type SwapchainCreateFlags = u32;
