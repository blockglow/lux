use std::alloc::Global;
use std::ops::Deref;

use crate::prelude::*;

use super::include::*;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Buffer(*const c_void);

#[repr(C)]
pub enum BufferCreateFlagBits {
    SparseBinding = 0x00000001,
    SparseResidency = 0x00000002,
    SparseAliased = 0x00000004,
    Protected = 0x00000008,
    DeviceAddressCaptureReplay = 0x00000010,
}

pub type BufferUsageFlags = u32;

#[repr(C)]
pub enum BufferUsageFlagBits {
    TransferSrc = 0x00000001,
    TransferDst = 0x00000002,
    UniformTexelBuffer = 0x00000004,
    StorageTexelBuffer = 0x00000008,
    UniformBuffer = 0x00000010,
    StorageBuffer = 0x00000020,
    IndexBuffer = 0x00000040,
    VertexBuffer = 0x00000080,
    IndirectBuffer = 0x00000100,
    ShaderDeviceAddress = 0x00020000,
    TransformFeedbackBufferExt = 0x00000800,
    TransformFeedbackCounterBufferExt = 0x00001000,
    ConditionalRenderingExt = 0x00000200,
    AccelerationStructureBuildInputReadOnlyKhr = 0x00080000,
    AccelerationStructureStorageKhr = 0x00100000,
    ShaderBindingTableKhr = 0x00000400,
}

pub type BufferCreateFlags = u32;

#[repr(C)]
pub struct BufferCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: BufferCreateFlags,
    pub size: DeviceSize,
    pub usage: BufferUsageFlags,
    pub sharing_mode: SharingMode,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

impl Default for Buffer {
    fn default() -> Self {
        Self(ptr::null())
    }
}

#[derive(Clone, Copy, Resource)]
pub struct GlobalBuffer {
    buffer: Buffer,
    memory: Memory,
}

impl Deref for GlobalBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl GlobalBuffer {
    pub(crate) fn new(
        physical_device: ResMut<PhysicalDevice>,
        device: ResMut<Device>,
    ) -> Insert<Self> {
        let buffer = Buffer::create(
            *device,
            8192,
            BufferUsageFlagBits::StorageBuffer as u32
                | BufferUsageFlagBits::TransferDst as u32
                | BufferUsageFlagBits::ShaderDeviceAddress as u32,
        );

        let req = buffer.requirements(*device);

        let memory =
            Memory::device_local(*physical_device, *device, req).bind_buffer(*device, buffer);

        (Self { buffer, memory }).into()
    }
}
