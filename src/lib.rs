#![feature(allocator_api, try_blocks)]
#![no_std]
extern crate alloc as core_alloc;

pub mod prelude {
        pub use core::result::*;
        pub use core_alloc::vec;

        pub use crate::boxed::*;
        pub use crate::collections::*;
        pub use crate::env::{error, info, trace, warn};
        pub use crate::ffi::*;
        pub use crate::fmt::*;
        pub use crate::string::*;
        pub use crate::vec::*;
        pub use crate::ver::*;

        pub fn default<T: Default>() -> T {
                Default::default()
        }
}

mod fmt {
        pub use core_alloc::fmt::*;
        pub use core_alloc::format;
}

mod collections {
        pub use core_alloc::collections::*;
}

mod vec {
        pub use core_alloc::vec::*;
}

mod string {
        pub use core_alloc::string::*;
}

mod boxed {
        pub use core_alloc::boxed::*;
}

mod ffi {
        pub use core_alloc::ffi::*;
}

mod alloc {
        use core::alloc;
        use core::alloc::Layout;

        pub struct Allocator;

        #[cfg(target_os = "linux")]
        unsafe impl alloc::GlobalAlloc for Allocator {
                unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                        libc::malloc(layout.size()) as *mut _
                }

                unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                        libc::free(ptr as *mut _);
                }
        }

        #[global_allocator]
        static ALLOCATOR: Allocator = Allocator;
}

pub mod env {
        use core::str::FromStr;

        pub use crate::prelude::*;

        pub struct Console;

        impl Console {
                pub fn log(level: Level, msg: &str) {
                        let mut string = msg.to_string();
                        string += "\n";
                        let mut c_string = CString::from_str(&string).unwrap();
                        unsafe { libc::write(1, c_string.as_ptr() as *const _, c_string.count_bytes()) };
                }
        }

        pub enum Level {
                Trace,
                Info,
                Warn,
                Error,
        }

        #[macro_export]
        macro_rules! log_trace {
                () => {
                        let output = String::new();
                        lux::env::Console::log(lux::env::Level::Trace, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        lux::env::Console::log(lux::env::Level::Trace, &output);
                }};
        }
        pub use log_trace as trace;

        #[macro_export]
        macro_rules! log_info {
                () => {
                        let output = String::new();
                        lux::env::Console::log(lux::env::Level::Info, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        lux::env::Console::log(lux::env::Level::Info, &output);
                }};
        }
        pub use log_info as info;

        #[macro_export]
        macro_rules! log_warn {
                () => {
                        let output = String::new();
                        lux::env::Console::log(lux::env::Level::Warn, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        lux::env::Console::log(lux::env::Level::Warn, &output);
                }};
        }
        pub use log_warn as warn;

        #[macro_export]
        macro_rules! log_error {
                () => {
                        let output = String::new();
                        lux::env::Console::log(lux::env::Level::Error, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        lux::env::Console::log(lux::env::Level::Error, &output);
                }};
        }
        pub use log_error as error;
}

pub mod gfx {
        use core::ffi::c_void;

        use crate::prelude::*;

        #[cfg(target_os = "linux")]
        pub mod vk {
                use core::{mem, ptr};
                use core::ffi::{c_void, CStr};
                use core::str::FromStr;

                use libc::c_char;

                use crate::env::{Console, Level};
                use crate::gfx::VkSurface;
                use crate::prelude::*;

                pub enum Error {
                        Unknown
                }

                pub struct ContextInfo {
                        app_name: String,
                        app_version: Version,
                }

                pub struct Context {
                        instance: *const c_void,
                        physical_device: *const c_void,
                        device: *const c_void,
                        surface: *const c_void,
                        swapchain: *const c_void,
                }

                pub trait ApiVersion {
                        fn make(self) -> u32;
                }

                impl ApiVersion for (u8, u8, u8) {
                        fn make(self) -> u32 {
                                let (a, b, c) = self;
                                let x = [0, c, b, a];
                                x.into_iter().enumerate().fold(0u32, |accum, (i, x)| {
                                        (x as u32) << (i as u32 * u8::BITS) | accum
                                })
                        }
                }

                fn c_str_array(array: &[impl AsRef<str>]) -> (Vec<CString>, Vec<*const c_char>, *const *const c_char) {
                        let c_strings = array.iter().map(AsRef::as_ref).map(CString::from_str).map(|x| x.unwrap()).collect::<Vec<_>>();
                        let c_ptrs = c_strings.iter().map(CString::as_ref).map(CStr::as_ptr).collect::<Vec<_>>();
                        let c_ptr_ptr = c_ptrs.as_ptr();
                        (c_strings, c_ptrs, c_ptr_ptr)
                }

                pub fn convert_c_str<const N: usize>(mut array: [u8; N]) -> String {
                        let mut v = array.to_vec();
                        v.truncate(array.iter().enumerate().find_map(|(i, &x)| (x == 0).then_some(i)).unwrap());
                        let string_c = unsafe { CString::from_vec_unchecked(v) };
                        string_c.to_str().unwrap().to_string()
                }

                impl Context {
                        pub fn acquire(surface: &dyn VkSurface) -> Result<Self, Error> {
                                let app_name = CString::from_str("yo").unwrap();
                                let engine_name = CString::from_str("yo").unwrap();
                                let ext_instance: Vec<&str> = vec!["VK_KHR_xlib_surface"];
                                let ext_device: Vec<&str> = vec![];
                                try {
                                        unsafe {
                                                let app_info = VkApplicationInfo {
                                                        s_type: VkStructureType::ApplicationInfo,
                                                        p_next: ptr::null(),
                                                        app_name: app_name.as_ptr(),
                                                        app_version: 0,
                                                        engine_name: engine_name.as_ptr(),
                                                        engine_version: 0,
                                                        api_version: (1, 3, 0).make(),
                                                };

                                                let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) = c_str_array(&*ext_instance);

                                                let instance_info = VkInstanceCreateInfo {
                                                        s_type: VkStructureType::InstanceCreateInfo,
                                                        p_next: ptr::null(),
                                                        flags: 0,
                                                        app_info: &app_info,
                                                        enabled_layer_count: 0,
                                                        enabled_layer: ptr::null(),
                                                        enabled_ext_count: enabled_ext_strings.len() as u32,
                                                        enabled_ext,
                                                };
                                                let mut instance = ptr::null();
                                                VkResult::handle(vkCreateInstance(&instance_info, ptr::null(), &mut instance))?;

                                                let mut count = 8;
                                                let mut physical_devices = vec![ptr::null(); count as usize];

                                                VkResult::handle(vkEnumeratePhysicalDevices(instance, &mut count, physical_devices.as_mut_ptr()))?;

                                                physical_devices.truncate(count as usize);

                                                let score = |property: VkPhysicalDeviceProperties| -> u64 {
                                                        let memory = property.limits.max_memory_allocation_count as u64;

                                                        let ty = match property.device_type {
                                                                VkPhysicalDeviceType::DiscreteGpu => u64::MAX / 2,
                                                                VkPhysicalDeviceType::IntegratedGpu => u64::MAX / 3,
                                                                VkPhysicalDeviceType::VirtualGpu => u64::MAX / 4,
                                                                VkPhysicalDeviceType::Cpu => u64::MAX / 5,
                                                                VkPhysicalDeviceType::Other => u64::MAX / 5,
                                                        };

                                                        memory + ty
                                                };

                                                let mut rank = BTreeMap::new();

                                                for physical_device in physical_devices {
                                                        let mut property = mem::zeroed::<VkPhysicalDeviceProperties>();
                                                        vkGetPhysicalDeviceProperties(physical_device, &mut property);

                                                        let name = convert_c_str(property.device_name);
                                                        Console::log(Level::Trace, &format!("Found GPU: \"{name}\""));

                                                        rank.insert((score)(property), (physical_device, name));
                                                }

                                                let (physical_device, physical_device_name) = rank.pop_last().map(|(_, x)| x).unwrap();
                                                Console::log(Level::Trace, &format!("Using GPU: \"{physical_device_name}\""));

                                                let mut count = 8;
                                                let mut queue_family_properties = vec![mem::zeroed::<VkQueueFamilyProperties>(); 8];
                                                vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut count, queue_family_properties.as_mut_ptr());

                                                let queue_family_index = queue_family_properties.iter()
                                                        .enumerate()
                                                        .find(|(_index, properties)| {
                                                                properties.queue_flags & QueueFlagBits::Graphics as u32 != 0
                                                        })
                                                        .map(|(index, _properties)| index as u32).unwrap();

                                                let enabled_features = VkPhysicalDeviceFeatures {
                                                        ..default()
                                                };

                                                let queue_priorities = [1.0];

                                                let queue_create_info = VkDeviceQueueCreateInfo {
                                                        s_type: VkStructureType::DeviceQueueInfo,
                                                        p_next: ptr::null(),
                                                        flags: 0,
                                                        queue_family_index,
                                                        queue_count: 1,
                                                        queue_priorities: queue_priorities.as_ptr(),
                                                };

                                                let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) = c_str_array(&*ext_device);

                                                let device_info = VkDeviceCreateInfo {
                                                        s_type: VkStructureType::DeviceInfo,
                                                        p_next: ptr::null(),
                                                        flags: 0,
                                                        queue_create_info_count: 1,
                                                        queue_create_info: &queue_create_info,
                                                        enabled_layer_count: 0,
                                                        enabled_layer: ptr::null(),
                                                        enabled_ext_count: enabled_ext_strings.len() as u32,
                                                        enabled_ext,
                                                        enabled_features: &enabled_features,
                                                };

                                                let mut device = ptr::null();
                                                VkResult::handle(vkCreateDevice(physical_device, &device_info, ptr::null(), &mut device))?;

                                                let surface = surface.to_vk_handle(instance)?;

                                                Self {
                                                        instance,
                                                        physical_device,
                                                        device,
                                                        surface,
                                                        swapchain: todo!(),
                                                }
                                        }
                                }
                        }
                }

                #[repr(u32)]
                pub enum VkStructureType {
                        ApplicationInfo = 0,
                        InstanceCreateInfo = 1,
                        DeviceQueueInfo = 2,
                        DeviceInfo = 3,
                        XlibSurfaceCreateInfo = 1000004000,
                }

                #[repr(C)]
                struct VkApplicationInfo {
                        s_type: VkStructureType,
                        p_next: *const c_void,
                        app_name: *const c_char,
                        app_version: u32,
                        engine_name: *const c_char,
                        engine_version: u32,
                        api_version: u32,
                }

                #[repr(C)]
                struct VkInstanceCreateInfo {
                        s_type: VkStructureType,
                        p_next: *const c_void,
                        flags: u64,
                        app_info: *const VkApplicationInfo,
                        enabled_layer_count: u32,
                        enabled_layer: *const *const c_char,
                        enabled_ext_count: u32,
                        enabled_ext: *const *const c_char,
                }

                pub type VkDeviceSize = u64;
                pub type VkBool32 = u32;
                pub type VkSampleCountFlags = u32;

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct VkPhysicalDeviceLimits {
                        pub max_image_dimension_1d: u32,
                        pub max_image_dimension_2d: u32,
                        pub max_image_dimension_3d: u32,
                        pub max_image_dimension_cube: u32,
                        pub max_image_array_layers: u32,
                        pub max_texel_buffer_elements: u32,
                        pub max_uniform_buffer_range: u32,
                        pub max_storage_buffer_range: u32,
                        pub max_push_constants_size: u32,
                        pub max_memory_allocation_count: u32,
                        pub max_sampler_allocation_count: u32,
                        pub buffer_image_granularity: VkDeviceSize,
                        pub sparse_address_space_size: VkDeviceSize,
                        pub max_bound_descriptor_sets: u32,
                        pub max_per_stage_descriptor_samplers: u32,
                        pub max_per_stage_descriptor_uniform_buffers: u32,
                        pub max_per_stage_descriptor_storage_buffers: u32,
                        pub max_per_stage_descriptor_sampled_images: u32,
                        pub max_per_stage_descriptor_storage_images: u32,
                        pub max_per_stage_descriptor_input_attachments: u32,
                        pub max_per_stage_resources: u32,
                        pub max_descriptor_set_samplers: u32,
                        pub max_descriptor_set_uniform_buffers: u32,
                        pub max_descriptor_set_uniform_buffers_dynamic: u32,
                        pub max_descriptor_set_storage_buffers: u32,
                        pub max_descriptor_set_storage_buffers_dynamic: u32,
                        pub max_descriptor_set_sampled_images: u32,
                        pub max_descriptor_set_storage_images: u32,
                        pub max_descriptor_set_input_attachments: u32,
                        pub max_vertex_input_attributes: u32,
                        pub max_vertex_input_bindings: u32,
                        pub max_vertex_input_attribute_offset: u32,
                        pub max_vertex_input_binding_stride: u32,
                        pub max_vertex_output_components: u32,
                        pub max_tessellation_generation_level: u32,
                        pub max_tessellation_patch_size: u32,
                        pub max_tessellation_control_per_vertex_input_components: u32,
                        pub max_tessellation_control_per_vertex_output_components: u32,
                        pub max_tessellation_control_per_patch_output_components: u32,
                        pub max_tessellation_control_total_output_components: u32,
                        pub max_tessellation_evaluation_input_components: u32,
                        pub max_tessellation_evaluation_output_components: u32,
                        pub max_geometry_shader_invocations: u32,
                        pub max_geometry_input_components: u32,
                        pub max_geometry_output_components: u32,
                        pub max_geometry_output_vertices: u32,
                        pub max_geometry_total_output_components: u32,
                        pub max_fragment_input_components: u32,
                        pub max_fragment_output_attachments: u32,
                        pub max_fragment_dual_src_attachments: u32,
                        pub max_fragment_combined_output_resources: u32,
                        pub max_compute_shared_memory_size: u32,
                        pub max_compute_work_group_count: [u32; 3],
                        pub max_compute_work_group_invocations: u32,
                        pub max_compute_work_group_size: [u32; 3],
                        pub sub_pixel_precision_bits: u32,
                        pub sub_texel_precision_bits: u32,
                        pub mipmap_precision_bits: u32,
                        pub max_draw_indexed_index_value: u32,
                        pub max_draw_indirect_count: u32,
                        pub max_sampler_lod_bias: f32,
                        pub max_sampler_anisotropy: f32,
                        pub max_viewports: u32,
                        pub max_viewport_dimensions: [u32; 2],
                        pub viewport_bounds_range: [f32; 2],
                        pub viewport_sub_pixel_bits: u32,
                        pub min_memory_map_alignment: usize,
                        pub min_texel_buffer_offset_alignment: VkDeviceSize,
                        pub min_uniform_buffer_offset_alignment: VkDeviceSize,
                        pub min_storage_buffer_offset_alignment: VkDeviceSize,
                        pub min_texel_offset: i32,
                        pub max_texel_offset: u32,
                        pub min_texel_gather_offset: i32,
                        pub max_texel_gather_offset: u32,
                        pub min_interpolation_offset: f32,
                        pub max_interpolation_offset: f32,
                        pub sub_pixel_interpolation_offset_bits: u32,
                        pub max_framebuffer_width: u32,
                        pub max_framebuffer_height: u32,
                        pub max_framebuffer_layers: u32,
                        pub framebuffer_color_sample_counts: VkSampleCountFlags,
                        pub framebuffer_depth_sample_counts: VkSampleCountFlags,
                        pub framebuffer_stencil_sample_counts: VkSampleCountFlags,
                        pub framebuffer_no_attachments_sample_counts: VkSampleCountFlags,
                        pub max_color_attachments: u32,
                        pub sampled_image_color_sample_counts: VkSampleCountFlags,
                        pub sampled_image_integer_sample_counts: VkSampleCountFlags,
                        pub sampled_image_depth_sample_counts: VkSampleCountFlags,
                        pub sampled_image_stencil_sample_counts: VkSampleCountFlags,
                        pub storage_image_sample_counts: VkSampleCountFlags,
                        pub max_sample_mask_words: u32,
                        pub timestamp_compute_and_graphics: VkBool32,
                        pub timestamp_period: f32,
                        pub max_clip_distances: u32,
                        pub max_cull_distances: u32,
                        pub max_combined_clip_and_cull_distances: u32,
                        pub discrete_queue_priorities: u32,
                        pub point_size_range: [f32; 2],
                        pub line_width_range: [f32; 2],
                        pub point_size_granularity: f32,
                        pub line_width_granularity: f32,
                        pub strict_lines: VkBool32,
                        pub standard_sample_locations: VkBool32,
                        pub optimal_buffer_copy_offset_alignment: VkDeviceSize,
                        pub optimal_buffer_copy_row_pitch_alignment: VkDeviceSize,
                        pub non_coherent_atom_size: VkDeviceSize,
                }

                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, Eq)]
                pub enum VkPhysicalDeviceType {
                        Other = 0,
                        IntegratedGpu = 1,
                        DiscreteGpu = 2,
                        VirtualGpu = 3,
                        Cpu = 4,
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct VkPhysicalDeviceSparseProperties {
                        pub residency_standard_2d_block_shape: VkBool32,
                        pub residency_standard_2d_multisample_block_shape: VkBool32,
                        pub residency_standard_3d_block_shape: VkBool32,
                        pub residency_aligned_mip_size: VkBool32,
                        pub residency_non_resident_strict: VkBool32,
                }

                pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
                pub const VK_UUID_SIZE: usize = 16;

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct VkPhysicalDeviceProperties {
                        pub api_version: u32,
                        pub driver_version: u32,
                        pub vendor_id: u32,
                        pub device_id: u32,
                        pub device_type: VkPhysicalDeviceType,
                        pub device_name: [u8; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
                        pub pipeline_cache_uuid: [u8; VK_UUID_SIZE],
                        pub limits: VkPhysicalDeviceLimits,
                        pub sparse_properties: VkPhysicalDeviceSparseProperties,
                }

                #[repr(C)]
                pub struct VkDeviceCreateInfo {
                        s_type: VkStructureType,
                        p_next: *const c_void,
                        flags: VkDeviceCreateFlags,
                        queue_create_info_count: u32,
                        queue_create_info: *const VkDeviceQueueCreateInfo,
                        // enabledLayerCount is deprecated and should not be used
                        enabled_layer_count: u32,
                        // ppEnabledLayerNames is deprecated and should not be used
                        enabled_layer: *const *const c_char,
                        enabled_ext_count: u32,
                        enabled_ext: *const *const c_char,
                        enabled_features: *const VkPhysicalDeviceFeatures,
                }

                pub type VkDeviceCreateFlags = u32;

                #[repr(C)]
                pub struct VkDeviceQueueCreateInfo {
                        s_type: VkStructureType,
                        p_next: *const c_void,
                        flags: VkDeviceQueueCreateFlags,
                        queue_family_index: u32,
                        queue_count: u32,
                        queue_priorities: *const f32,
                }


                #[repr(C)]
                #[derive(Default)]
                pub struct VkPhysicalDeviceFeatures {
                        pub robust_buffer_access: VkBool32,
                        pub full_draw_index_uint32: VkBool32,
                        pub image_cube_array: VkBool32,
                        pub independent_blend: VkBool32,
                        pub geometry_shader: VkBool32,
                        pub tessellation_shader: VkBool32,
                        pub sample_rate_shading: VkBool32,
                        pub dual_src_blend: VkBool32,
                        pub logic_op: VkBool32,
                        pub multi_draw_indirect: VkBool32,
                        pub draw_indirect_first_instance: VkBool32,
                        pub depth_clamp: VkBool32,
                        pub depth_bias_clamp: VkBool32,
                        pub fill_mode_non_solid: VkBool32,
                        pub depth_bounds: VkBool32,
                        pub wide_lines: VkBool32,
                        pub large_points: VkBool32,
                        pub alpha_to_one: VkBool32,
                        pub multi_viewport: VkBool32,
                        pub sampler_anisotropy: VkBool32,
                        pub texture_compression_etc2: VkBool32,
                        pub texture_compression_astc_ldr: VkBool32,
                        pub texture_compression_bc: VkBool32,
                        pub occlusion_query_precise: VkBool32,
                        pub pipeline_statistics_query: VkBool32,
                        pub vertex_pipeline_stores_and_atomics: VkBool32,
                        pub fragment_stores_and_atomics: VkBool32,
                        pub shader_tessellation_and_geometry_point_size: VkBool32,
                        pub shader_image_gather_extended: VkBool32,
                        pub shader_storage_image_extended_formats: VkBool32,
                        pub shader_storage_image_multisample: VkBool32,
                        pub shader_storage_image_read_without_format: VkBool32,
                        pub shader_storage_image_write_without_format: VkBool32,
                        pub shader_uniform_buffer_array_dynamic_indexing: VkBool32,
                        pub shader_sampled_image_array_dynamic_indexing: VkBool32,
                        pub shader_storage_buffer_array_dynamic_indexing: VkBool32,
                        pub shader_storage_image_array_dynamic_indexing: VkBool32,
                        pub shader_clip_distance: VkBool32,
                        pub shader_cull_distance: VkBool32,
                        pub shader_float64: VkBool32,
                        pub shader_int64: VkBool32,
                        pub shader_int16: VkBool32,
                        pub shader_resource_residency: VkBool32,
                        pub shader_resource_min_lod: VkBool32,
                        pub sparse_binding: VkBool32,
                        pub sparse_residency_buffer: VkBool32,
                        pub sparse_residency_image2_d: VkBool32,
                        pub sparse_residency_image3_d: VkBool32,
                        pub sparse_residency2_samples: VkBool32,
                        pub sparse_residency4_samples: VkBool32,
                        pub sparse_residency8_samples: VkBool32,
                        pub sparse_residency16_samples: VkBool32,
                        pub sparse_residency_aliased: VkBool32,
                        pub variable_multisample_rate: VkBool32,
                        pub inherited_queries: VkBool32,
                }

                pub type VkQueueFlags = u32;

                #[repr(C)]
                #[derive(Debug, Clone, Copy)]
                pub struct VkExtent3D {
                        pub width: u32,
                        pub height: u32,
                        pub depth: u32,
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
                pub struct VkQueueFamilyProperties {
                        pub queue_flags: VkQueueFlags,
                        pub queue_count: u32,
                        pub timestamp_valid_bits: u32,
                        pub min_image_transfer_granularity: VkExtent3D,
                }

                pub type VkDeviceQueueCreateFlags = u32;

                #[repr(C)]
                pub struct VkXlibSurfaceCreateInfo {
                        pub s_type: VkStructureType,
                        pub p_next: *const c_void,
                        pub flags: u32,
                        pub display: *const c_void,
                        pub window: *const c_void,
                }

                #[repr(i32)]
                #[derive(Debug, Copy, Clone, PartialEq, Eq)]
                pub enum VkResult {
                        Success = 0,
                        NotReady = 1,
                        Timeout = 2,
                        EventSet = 3,
                        EventReset = 4,
                        Incomplete = 5,
                        ErrorOutOfHostMemory = -1,
                        ErrorOutOfDeviceMemory = -2,
                        ErrorInitializationFailed = -3,
                        ErrorDeviceLost = -4,
                        ErrorMemoryMapFailed = -5,
                        ErrorLayerNotPresent = -6,
                        ErrorExtensionNotPresent = -7,
                        ErrorFeatureNotPresent = -8,
                        ErrorIncompatibleDriver = -9,
                        ErrorTooManyObjects = -10,
                        ErrorFormatNotSupported = -11,
                        ErrorFragmentedPool = -12,
                        ErrorUnknown = -13,
                }

                impl VkResult {
                        pub(crate) fn handle(r: VkResult) -> Result<(), Error> {
                                match r {
                                        VkResult::Success => Ok(()),
                                        e => panic!("{e:?}"),
                                }
                        }
                }

                extern "C" {
                        fn vkCreateInstance(info: *const VkInstanceCreateInfo, alloc: *const c_void, out: *mut *const c_void) -> VkResult;
                        fn vkEnumeratePhysicalDevices(instance: *const c_void, count: *mut u32, physical_devices: *mut *const c_void) -> VkResult;
                        fn vkGetPhysicalDeviceProperties(physical_device: *const c_void, property: *mut VkPhysicalDeviceProperties);
                        fn vkCreateDevice(physical_device: *const c_void, info: *const VkDeviceCreateInfo, alloc: *const c_void, out: *mut *const c_void) -> VkResult;
                        fn vkGetPhysicalDeviceQueueFamilyProperties(physical_device: *const c_void, count: *mut u32, physical_devices: *mut VkQueueFamilyProperties) -> VkResult;
                        pub(crate) fn vkCreateXlibSurfaceKHR(instance: *const c_void, info: *const VkXlibSurfaceCreateInfo, alloc: *const c_void, surface: *mut *const c_void) -> VkResult;
                }
        }

        #[cfg(target_os = "linux")]
        mod x11 {
                use core::ffi::c_void;
                use core::ptr;

                use libc::c_char;

                use crate::gfx::{Surface, VkSurface};

                pub struct Window {
                        display: *const c_void,
                        window: *const c_void,
                }

                impl Surface for Window {
                        fn open() -> Self
                        where
                                Self: Sized,
                        {
                                let (display, window) = unsafe {
                                        let display = XOpenDisplay(ptr::null());

                                        let screen = XDefaultScreen(display);

                                        let window = XCreateSimpleWindow(
                                                display,
                                                XRootWindow(display, screen),
                                                10,
                                                10,
                                                100,
                                                100,
                                                1,
                                                XBlackPixel(display, screen),
                                                XWhitePixel(display, screen),
                                        );
                                        XMapWindow(display, window);

                                        (display, window)
                                };
                                Self {
                                        display,
                                        window,
                                }
                        }

                        fn update(&self) {
                                let mut buf = crate::core_alloc::vec![0u8; 65535];
                                unsafe {
                                        XNextEvent(self.display, buf.as_mut_ptr() as *mut _);
                                }
                        }
                }

                extern "C" {
                        fn XOpenDisplay(display_name: *const c_char) -> *const c_void;
                        fn XBlackPixel(display: *const c_void, screen: i32) -> u64;
                        fn XWhitePixel(display: *const c_void, screen: i32) -> u64;
                        fn XDefaultScreen(display: *const c_void) -> i32;
                        fn XRootWindow(display: *const c_void, screen: i32) -> *const c_void;
                        fn XMapWindow(display: *const c_void, window: *const c_void);
                        fn XCreateSimpleWindow(
                                display: *const c_void,
                                root: *const c_void,
                                x: i32,
                                y: i32,
                                width: u32,
                                height: u32,
                                border_width: u32,
                                border: u64,
                                background: u64,
                        ) -> *const c_void;
                        fn XNextEvent(display: *const c_void, event: *mut c_void);
                }

                impl VkSurface for Window {
                        fn to_vk_handle(&self, instance: *const c_void) -> Result<*const c_void, crate::gfx::vk::Error> {
                                use crate::gfx::vk::*;
                                let surface_info = VkXlibSurfaceCreateInfo {
                                        s_type: VkStructureType::XlibSurfaceCreateInfo,
                                        p_next: ptr::null(),
                                        flags: 0,
                                        display: self.display,
                                        window: self.window,
                                };

                                let mut surface = ptr::null();
                                (unsafe {
                                        VkResult::handle(vkCreateXlibSurfaceKHR(instance, &surface_info, ptr::null(), &mut surface))
                                }).map(|_| surface)
                        }
                }
        }


        #[cfg(target_os = "linux")]
        pub trait VkSurface: Surface {
                fn to_vk_handle(&self, instance: *const c_void) -> Result<*const c_void, crate::gfx::vk::Error>;
        }

        #[cfg(target_os = "linux")]
        pub fn make_vk_surface() -> Box<dyn VkSurface> {
                Box::new(x11::Window::open())
        }

        pub trait Surface {
                fn open() -> Self
                where
                        Self: Sized;
                fn update(&self);
        }
}

pub mod ver {
        #[repr(C)]
        pub struct Version {
                major: u8,
                minor: u8,
                patch: u8,
        }
}
