use super::include::*;

#[repr(C)]
pub struct MemoryRequirements {
    pub size: DeviceSize,
    pub alignment: DeviceSize,
    pub memory_type_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Memory(*const c_void);
impl Default for Memory {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl Memory {
    pub(crate) fn map(self, device: Device, offset: u64, size: u64) -> *mut c_void {
        let mut ptr = ptr::null_mut();
        unsafe { vkMapMemory(device, self, offset, size, 0, &mut ptr) };
        ptr
    }
    pub(crate) fn unmap(self, device: Device) {
        unsafe { vkUnmapMemory(device, self) };
    }
    pub(crate) fn host(
        physical_device: PhysicalDevice,
        device: Device,
        req: MemoryRequirements,
    ) -> Self {
        let info = MemoryAllocateInfo {
            s_type: StructureType::MemoryAllocateInfo,
            p_next: ptr::null(),
            allocation_size: req.size,
            memory_type_index: find_memory_type(
                physical_device,
                req.memory_type_bits,
                MemoryPropertyFlagBits::HostCoherent as u32
                    | MemoryPropertyFlagBits::HostVisible as u32,
            )
            .unwrap(),
        };
        let mut mem = Self::default();
        unsafe { vkAllocateMemory(device, &info, ptr::null(), &mut mem) };
        mem
    }
    pub(crate) fn device_local(
        physical_device: PhysicalDevice,
        device: Device,
        req: MemoryRequirements,
    ) -> Self {
        let flags = MemoryAllocateFlagsInfo {
            s_type: StructureType::MemoryAllocateFlagsInfo,
            p_next: ptr::null(),
            flags: MemoryAllocateFlagBits::DeviceAddress as u32,
            device_mask: 0,
        };

        let info = MemoryAllocateInfo {
            s_type: StructureType::MemoryAllocateInfo,
            p_next: &flags as *const _ as *const _,
            allocation_size: req.size,
            memory_type_index: find_memory_type(
                physical_device,
                req.memory_type_bits,
                MemoryPropertyFlagBits::DeviceLocal as u32,
            )
            .unwrap(),
        };
        let mut mem = Self::default();
        unsafe { vkAllocateMemory(device, &info, ptr::null(), &mut mem) };
        mem
    }

    pub(crate) fn bind_buffer(self, device: Device, buffer: Buffer) -> Self {
        unsafe { vkBindBufferMemory(device, buffer, self, 0) };
        self
    }
}

#[repr(C)]
pub struct MemoryAllocateInfo {
    s_type: StructureType,
    p_next: *const c_void,
    allocation_size: DeviceSize,
    memory_type_index: u32,
}

#[repr(C)]
pub struct MemoryAllocateFlagsInfo {
    s_type: StructureType,
    p_next: *const c_void,
    flags: MemoryAllocateFlags,
    device_mask: u32,
}

pub type MemoryAllocateFlags = u32;

#[repr(C)]
pub enum MemoryAllocateFlagBits {
    DeviceMask = 0x00000001,
    DeviceAddress = 0x00000002,
    DeviceAddressCaptureReplay = 0x00000004,
}

// Constants
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

#[repr(C)]
pub struct MemoryType {
    property_flags: MemoryPropertyFlags,
    heap_index: u32,
}

fn find_memory_type(
    physical_device: PhysicalDevice,
    filter: u32,
    properties: MemoryPropertyFlags,
) -> Option<u32> {
    let mut mem_properties = unsafe { mem::zeroed::<PhysicalDeviceMemoryProperties>() };
    unsafe { vkGetPhysicalDeviceMemoryProperties(physical_device, &mut mem_properties) };
    (0..mem_properties.memory_type_count as usize)
        .find(|i| {
            (filter & (1 << i) != 0)
                && (mem_properties.memory_types[*i].property_flags & properties == properties)
        })
        .map(|x| x as u32)
}

#[repr(C)]
pub struct MemoryHeap {
    size: DeviceSize,
    flags: MemoryHeapFlags,
}

pub type MemoryPropertyFlags = u32;

#[repr(u32)]
pub enum MemoryPropertyFlagBits {
    DeviceLocal = 0x00000001,
    HostVisible = 0x00000002,
    HostCoherent = 0x00000004,
    HostCached = 0x00000008,
    LazilyAllocated = 0x00000010,
    Protected = 0x00000020,
    DeviceCoherentAMD = 0x00000040,
    DeviceUncachedAMD = 0x00000080,
    RdmaCapableNV = 0x00000100,
}

pub type MemoryHeapFlags = u32;

#[repr(u32)]
pub enum MemoryHeapFlagBits {
    DeviceLocal = 0x00000001,
    MultiInstance = 0x00000002,
}
