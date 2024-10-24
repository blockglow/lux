use crate::prelude::*;

use super::include::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Device(*const c_void);
impl Default for Device {
    fn default() -> Self {
        Self(ptr::null())
    }
}

impl Device {
    pub(crate) fn new(
        physical_device: PhysicalDevice,
        surface: Surface,
    ) -> Result<(Self, QueueFamilyIndex), Error> {
        let ext_device: Vec<&str> = vec!["VK_KHR_swapchain", "VK_EXT_shader_object"];

        let buffer_device_address_features = PhysicalDeviceBufferDeviceAddressFeatures {
            s_type: StructureType::PhysicalDeviceBufferDeviceAddressFeatures,
            p_next: ptr::null(),
            buffer_device_address: true as _,
            buffer_device_address_capture_replay: false as _,
            buffer_device_address_multi_device: false as _,
        };

        let dynamic_rendering_features = PhysicalDeviceDynamicRenderingFeatures {
            s_type: StructureType::PhysicalDeviceDynamicRenderingFeatures,
            p_next: &buffer_device_address_features as *const _ as *const _,
            dynamic_rendering: true as _,
        };

        let shader_object_features = PhysicalDeviceShaderObjectFeatures {
            s_type: StructureType::PhysicalDeviceShaderObjectFeatures,
            p_next: &dynamic_rendering_features as *const _ as *const _,
            shader_object: true as _,
        };

        let indexing_features = PhysicalDeviceDescriptorIndexingFeatures {
            s_type: StructureType::PhysicalDeviceDescriptorIndexingFeatures,
            p_next: &shader_object_features as *const _ as *const _,
            shader_input_attachment_array_dynamic_indexing: 0,
            shader_uniform_texel_buffer_array_dynamic_indexing: 0,
            shader_storage_texel_buffer_array_dynamic_indexing: 0,
            shader_uniform_buffer_array_non_uniform_indexing: 0,
            shader_sampled_image_array_non_uniform_indexing: 0,
            shader_storage_buffer_array_non_uniform_indexing: 0,
            shader_storage_image_array_non_uniform_indexing: 0,
            shader_input_attachment_array_non_uniform_indexing: 0,
            shader_uniform_texel_buffer_array_non_uniform_indexing: 0,
            shader_storage_texel_buffer_array_non_uniform_indexing: 0,
            descriptor_binding_uniform_buffer_update_after_bind: 0,
            descriptor_binding_sampled_image_update_after_bind: 0,
            descriptor_binding_storage_image_update_after_bind: true as _,
            descriptor_binding_storage_buffer_update_after_bind: true as _,
            descriptor_binding_uniform_texel_buffer_update_after_bind: 0,
            descriptor_binding_storage_texel_buffer_update_after_bind: 0,
            descriptor_binding_update_unused_while_pending: 0,
            descriptor_binding_partially_bound: true as _,
            descriptor_binding_variable_descriptor_count: true as _,
            runtime_descriptor_array: 0,
        };

        let enabled_features = PhysicalDeviceFeatures2 {
            s_type: StructureType::PhysicalDeviceFeatures2,
            p_next: &indexing_features as *const _ as *const _,
            features: PhysicalDeviceFeatures { ..default() },
        };

        let queue_priorities = [1.0];
        let queue_family_index = QueueFamilyIndex::graphics(physical_device, surface)?;
        let queue_create_info = DeviceQueueCreateInfo {
            s_type: StructureType::DeviceQueueInfo,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index,
            queue_count: 1,
            queue_priorities: queue_priorities.as_ptr(),
        };

        let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) = c_str_array(&*ext_device);

        let device_info = DeviceCreateInfo {
            s_type: StructureType::DeviceInfo,
            p_next: &enabled_features as *const _ as *const _,
            flags: 0,
            queue_create_info_count: 1,
            queue_create_info: &queue_create_info,
            enabled_layer_count: 0,
            enabled_layer: ptr::null(),
            enabled_ext_count: enabled_ext_strings.len() as u32,
            enabled_ext,
            enabled_features: ptr::null(),
        };

        let mut device = Device::default();
        VkResult::handle(unsafe {
            vkCreateDevice(physical_device, &device_info, ptr::null(), &mut device)
        })?;

        Ok((device, queue_family_index))
    }
}

#[repr(C)]
pub struct DeviceCreateInfo {
    s_type: StructureType,
    p_next: *const c_void,
    flags: DeviceCreateFlags,
    queue_create_info_count: u32,
    queue_create_info: *const DeviceQueueCreateInfo,
    // enabledLayerCount is deprecated and should not be used
    enabled_layer_count: u32,
    // ppEnabledLayerNames is deprecated and should not be used
    enabled_layer: *const *const c_char,
    enabled_ext_count: u32,
    enabled_ext: *const *const c_char,
    enabled_features: *const PhysicalDeviceFeatures,
}

pub type DeviceCreateFlags = u32;
pub type QueueFlags = u32;

#[repr(C)]
pub struct DeviceQueueCreateInfo {
    s_type: StructureType,
    p_next: *const c_void,
    flags: DeviceQueueCreateFlags,
    queue_family_index: QueueFamilyIndex,
    queue_count: u32,
    queue_priorities: *const f32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueFlagBits {
    Graphics = 0x00000001,
    Compute = 0x00000002,
    Transfer = 0x00000004,
    SparseBinding = 0x00000008,
    Protected = 0x00000010,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QueueFamilyProperties {
    pub queue_flags: QueueFlags,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: Extent3D,
}

pub type DeviceQueueCreateFlags = u32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Queue(*const c_void);

impl Default for Queue {
    fn default() -> Self {
        Self(ptr::null())
    }
}

impl Queue {
    pub fn submit(self, wait: Semaphore, signal: Semaphore, fence: Fence, cmd: CommandBuffer) {
        let stage_mask = [PipelineStageFlagBits::ColorAttachmentOutput as u32];
        let info = SubmitInfo {
            s_type: StructureType::SubmitInfo,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: &wait,
            p_wait_dst_stage_mask: stage_mask.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            signal_semaphore_count: 1,
            p_signal_semaphores: &signal,
        };
        VkResult::handle(unsafe { vkQueueSubmit(self, 1, &info, fence) }).unwrap();
    }

    pub fn present(self, wait: Semaphore, swapchain: Swapchain, index: u32) {
        let mut result = VkResult::Success;
        let info = PresentInfo {
            s_type: StructureType::PresentInfo,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: &wait,
            swapchain_count: 1,
            p_swapchains: &swapchain,
            p_image_indices: &index,
            p_results: &mut result,
        };
        unsafe { vkQueuePresentKHR(self, &info) };
        VkResult::handle(result).unwrap();
    }
}
