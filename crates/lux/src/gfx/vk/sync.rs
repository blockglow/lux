use std::ops::Index;

use libc::if_indextoname;

use crate::prelude::*;

use super::include::*;

#[repr(C)]
pub struct SemaphoreCreateInfo {
    s_type: StructureType,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
pub struct FenceCreateInfo {
    s_type: StructureType,
    p_next: *const c_void,
    flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Fence(*const c_void);
impl Default for Fence {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl Fence {
    pub(crate) fn create_signaled(device: Device) -> Result<Self, Error> {
        let info = FenceCreateInfo {
            s_type: StructureType::FenceCreateInfo,
            p_next: ptr::null(),
            flags: 1,
        };
        let mut fence = Self::default();
        unsafe { vkCreateFence(device, &info, ptr::null(), &mut fence) };
        Ok(fence)
    }
    fn create(device: Device) -> Result<Self, Error> {
        let info = FenceCreateInfo {
            s_type: StructureType::FenceCreateInfo,
            p_next: ptr::null(),
            flags: 0,
        };
        let mut fence = Self::default();
        unsafe { vkCreateFence(device, &info, ptr::null(), &mut fence) };
        Ok(fence)
    }
    pub(crate) fn wait(self, device: Device) -> bool {
        VkResult::handle(unsafe { vkWaitForFences(device, 1, &self, true as _, 0) }).is_err()
    }
    pub(crate) fn reset(self, device: Device) {
        unsafe { vkResetFences(device, 1, &self) };
    }
}

#[repr(C)]
#[derive(Clone, Copy, Resource)]
pub struct Semaphore(*const c_void);
impl Default for Semaphore {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl Semaphore {
    pub(crate) fn create(device: Device) -> Result<Self, Error> {
        let info = SemaphoreCreateInfo {
            s_type: StructureType::SemaphoreCreateInfo,
            p_next: ptr::null(),
            flags: 0,
        };
        let mut semaphore = Self::default();
        info!("YOOO");
        unsafe { vkCreateSemaphore(device, &info, ptr::null(), &mut semaphore) };
        Ok(semaphore)
    }
}

pub struct FrameSyncHandles {
    pub(crate) image_avail: Semaphore,
    pub(crate) render_finish: Semaphore,
    pub(crate) ready: Fence,
}

#[derive(Resource)]
pub struct FrameSync(Vec<FrameSyncHandles>);

impl FrameSync {
    pub(crate) fn new(device: ResMut<Device>) -> Insert<Self> {
        FrameSync(
            iter::repeat_with(|| FrameSyncHandles {
                image_avail: Semaphore::create(*device).unwrap(),
                render_finish: Semaphore::create(*device).unwrap(),
                ready: Fence::create_signaled(*device).unwrap(),
            })
            .take(4)
            .collect(),
        )
        .into()
    }
}

impl Index<usize> for FrameSync {
    type Output = FrameSyncHandles;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
