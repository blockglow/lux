use crate::gfx::ActiveSurface;
use crate::prelude::*;

use super::include::*;

#[derive(Clone, Copy, Resource, Default)]
pub struct FrameIndex(usize);

impl ops::Deref for FrameIndex {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ops::DerefMut for FrameIndex {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Resource)]
pub struct FrameImages {
    pub(crate) images: Vec<Image>,
    pub(crate) image_views: Vec<ImageView>,
}

#[repr(C)]
#[derive(Clone, Copy, Resource)]
pub struct Swapchain(*const c_void);
impl Default for Swapchain {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl Swapchain {
    pub(crate) fn next_image(
        self,
        device: Device,
        semaphore: Semaphore,
    ) -> Result<FrameIndex, Error> {
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
        Ok(FrameIndex(index as usize))
    }
    pub(crate) fn images(
        device: ResMut<Device>,
        swapchain: ResMut<Swapchain>,
    ) -> Insert<FrameImages> {
        let mut count = 8;
        let mut images = vec![Image::default(); count as usize];

        unsafe { vkGetSwapchainImagesKHR(*device, *swapchain, &mut count, images.as_mut_ptr()) }

        images.truncate(count as usize);

        let image_views = images
            .iter()
            .map(|image| {
                ImageView::create(*device, *image, ImageViewType::Type2d, Format::Bgra8Unorm)
                    .unwrap()
            })
            .collect();

        (FrameImages {
            images,
            image_views,
        })
        .into()
    }
    pub(crate) fn new(
        physical_device: ResMut<PhysicalDevice>,
        device: ResMut<Device>,
        queue_family_index: ResMut<QueueFamilyIndex>,
        surface: ResMut<Surface>,
        old_swapchain: Option<ResMut<Swapchain>>,
    ) -> Insert<Swapchain> {
        let Extent2D { width, height } = surface.capabilities(*physical_device).current_extent;
        let swapchain_info = SwapchainCreateInfo {
            s_type: StructureType::SwapchainCreateInfo,
            p_next: ptr::null(),
            flags: 0,
            surface: *surface,
            min_image_count: 3,
            image_format: Format::Bgra8Unorm,
            image_color_space: ColorSpace::SrgbNonlinear,
            image_extent: Extent2D { width, height },
            image_array_layers: 1,
            image_usage: ImageUsageFlagBits::ColorAttachment as u32,
            image_sharing_mode: SharingMode::Exclusive,
            queue_family_index_count: 1,
            p_queue_family_indices: &*queue_family_index,
            pre_transform: SurfaceTransformFlagBits::IdentityBit,
            composite_alpha: CompositeAlphaFlagBits::OpaqueBit,
            present_mode: PresentMode::Immediate,
            clipped: 0,
            old_swapchain: old_swapchain.map(|x| *x),
        };

        let mut swapchain = Swapchain::default();
        VkResult::handle(dbg!(unsafe {
            vkCreateSwapchainKHR(*device, &swapchain_info, ptr::null(), &mut swapchain)
        }))
        .unwrap();

        swapchain.into()
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
