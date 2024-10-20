#![feature(allocator_api, try_blocks)]
#![no_std]
extern crate alloc as core_alloc;

pub mod prelude {
        pub use core::result::*;
        pub use core_alloc::vec;

        pub use crate::boxed::*;
        pub use crate::collections::*;
        pub use crate::env::{dbg, error, info, trace, warn};
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
        macro_rules! log_dbg {
                ($($arg:tt)*) => {{
                        let var = $($arg)*;
                        let output = format!("{:?}", &var);
                        $crate::env::Console::log($crate::env::Level::Trace, &output);
                        var
                }};
        }
        pub use log_dbg as dbg;

        #[macro_export]
        macro_rules! log_trace {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Trace, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Trace, &output);
                }};
        }
        pub use log_trace as trace;

        #[macro_export]
        macro_rules! log_info {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Info, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Info, &output);
                }};
        }
        pub use log_info as info;

        #[macro_export]
        macro_rules! log_warn {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Warn, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Warn, &output);
                }};
        }
        pub use log_warn as warn;

        #[macro_export]
        macro_rules! log_error {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Error, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Error, &output);
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
                use crate::gfx::{Renderer as GfxRenderer, Surface as GfxSurface, SurfaceHandle};
                use crate::prelude::*;

                pub struct Renderer {
                        instance: Instance,
                        physical_device: PhysicalDevice,
                        device: Device,
                        qfi: QueueFamilyIndex,
                        surface: Surface,
                        swapchain: Swapchain,
                        descriptor_table: DescriptorTable,
                        descriptor_pool: DescriptorPool,
                        descriptor_set_layout: DescriptorSetLayout,
                        descriptor_sets: DescriptorSets,
                }

                impl GfxRenderer for Renderer {
                        fn create(surface: &dyn GfxSurface) -> Self
                        where
                                Self: Sized,
                        {
                                let instance = Instance::new().unwrap();
                                let physical_device = PhysicalDevice::acquire_best(instance).unwrap();
                                let (device, qfi) = Device::new(physical_device).unwrap();
                                let surface = Surface::open(instance, surface).unwrap();
                                let swapchain = Swapchain::new(device, qfi, surface, None).unwrap();
                                let descriptor_table = DescriptorTable::default();
                                let descriptor_pool = DescriptorPool::bindless(device, &descriptor_table).unwrap();
                                let descriptor_set_layout = DescriptorSetLayout::bindless(device, &descriptor_table).unwrap();
                                let descriptor_sets = DescriptorSets::allocate(device, descriptor_pool, descriptor_set_layout).unwrap();

                                Self {
                                        instance,
                                        physical_device,
                                        device,
                                        qfi,
                                        surface,
                                        swapchain,
                                        descriptor_table,
                                        descriptor_pool,
                                        descriptor_set_layout,
                                        descriptor_sets,
                                }
                        }

                        fn render(&mut self) {}
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct Instance(*const c_void);

                impl Default for Instance {
                        fn default() -> Self {
                                Instance(ptr::null())
                        }
                }

                impl Instance {
                        fn new() -> Result<Self, Error> {
                                let app_name = CString::from_str("yo").unwrap();
                                let engine_name = CString::from_str("yo").unwrap();
                                let mut ext_instance: Vec<&str> = vec!["VK_KHR_surface"];
                                #[cfg(target_os = "linux")]
                                ext_instance.push("VK_KHR_xlib_surface");
                                let app_info = ApplicationInfo {
                                        s_type: StructureType::ApplicationInfo,
                                        p_next: ptr::null(),
                                        app_name: app_name.as_ptr(),
                                        app_version: 0,
                                        engine_name: engine_name.as_ptr(),
                                        engine_version: 0,
                                        api_version: (1, 3, 0).make(),
                                };

                                let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) = c_str_array(&*ext_instance);

                                let instance_info = InstanceCreateInfo {
                                        s_type: StructureType::InstanceCreateInfo,
                                        p_next: ptr::null(),
                                        flags: 0,
                                        app_info: &app_info,
                                        enabled_layer_count: 0,
                                        enabled_layer: ptr::null(),
                                        enabled_ext_count: enabled_ext_strings.len() as u32,
                                        enabled_ext,
                                };
                                let mut instance = Self::default();
                                VkResult::handle(unsafe { vkCreateInstance(&instance_info, ptr::null(), &mut instance) })?;
                                Ok(instance)
                        }
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct PhysicalDevice(*const c_void);
                impl Default for PhysicalDevice {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }
                impl PhysicalDevice {
                        fn acquire_best(instance: Instance) -> Result<Self, Error> {
                                let mut count = 8;
                                let mut physical_devices = vec![PhysicalDevice::default(); count as usize];

                                VkResult::handle(unsafe { vkEnumeratePhysicalDevices(instance, &mut count, physical_devices.as_mut_ptr()) })?;

                                physical_devices.truncate(count as usize);

                                let score = |property: PhysicalDeviceProperties| -> u64 {
                                        let memory = property.limits.max_memory_allocation_count as u64;

                                        let ty = match property.device_type {
                                                PhysicalDeviceType::DiscreteGpu => u64::MAX / 2,
                                                PhysicalDeviceType::IntegratedGpu => u64::MAX / 3,
                                                PhysicalDeviceType::VirtualGpu => u64::MAX / 4,
                                                PhysicalDeviceType::Cpu => u64::MAX / 5,
                                                PhysicalDeviceType::Other => u64::MAX / 5,
                                        };

                                        memory + ty
                                };

                                let mut rank = BTreeMap::new();

                                for physical_device in physical_devices {
                                        let mut property = unsafe { mem::zeroed::<PhysicalDeviceProperties>() };
                                        unsafe {
                                                vkGetPhysicalDeviceProperties(physical_device, &mut property);
                                        }

                                        let name = convert_c_str(property.device_name);
                                        Console::log(Level::Trace, &format!("Found GPU: \"{name}\""));

                                        rank.insert((score)(property), (physical_device, name));
                                }

                                let (physical_device, physical_device_name) = rank.pop_last().map(|(_, x)| x).unwrap();
                                Console::log(Level::Trace, &format!("Using GPU: \"{physical_device_name}\""));

                                Ok(physical_device)
                        }
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct Device(*const c_void);
                impl Default for Device {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }

                #[repr(C)]
                pub struct PhysicalDeviceFeatures2 {
                        s_type: StructureType,
                        p_next: *const c_void,
                        features: PhysicalDeviceFeatures,
                }

                impl Device {
                        fn new(physical_device: PhysicalDevice) -> Result<(Self, QueueFamilyIndex), Error> {
                                let ext_device: Vec<&str> = vec!["VK_KHR_swapchain"];

                                let indexing_features = PhysicalDeviceDescriptorIndexingFeatures {
                                        s_type: StructureType::PhysicalDeviceDescriptorIndexingFeatures,
                                        p_next: ptr::null(),
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
                                let queue_family_index = QueueFamilyIndex::graphics(physical_device)?;
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
                                VkResult::handle(unsafe { vkCreateDevice(physical_device, &device_info, ptr::null(), &mut device) })?;

                                Ok((device, queue_family_index))
                        }
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct Surface(*const c_void);
                impl Default for Surface {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }
                impl Surface {
                        fn open(instance: Instance, gfx: &dyn GfxSurface) -> Result<Self, Error> {
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
                                VkResult::handle(unsafe { vkCreateXlibSurfaceKHR(instance, &surface_info, ptr::null(), &mut surface) })?;

                                Ok(surface)
                        }
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct QueueFamilyIndex(u32);

                impl QueueFamilyIndex {
                        fn graphics(physical_device: PhysicalDevice) -> Result<Self, Error> {
                                let mut count = 8;
                                let mut queue_family_properties = vec![unsafe { mem::zeroed::<QueueFamilyProperties>() }; 8];
                                unsafe { vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &mut count, queue_family_properties.as_mut_ptr()) };

                                let queue_family_index = queue_family_properties.iter()
                                        .enumerate()
                                        .find(|(_index, properties)| {
                                                properties.queue_flags & QueueFlagBits::Graphics as u32 != 0
                                        })
                                        .map(|(index, _properties)| index as u32).unwrap();

                                Ok(Self(queue_family_index))
                        }
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct Swapchain(*const c_void);
                impl Default for Swapchain {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }
                impl Swapchain {
                        fn new(device: Device, queue_family_index: QueueFamilyIndex, surface: Surface, old_swapchain: Option<Swapchain>) -> Result<Swapchain, Error> {
                                let swapchain_info = SwapchainCreateInfo {
                                        s_type: StructureType::SwapchainCreateInfo,
                                        p_next: ptr::null(),
                                        flags: 0,
                                        surface,
                                        min_image_count: 3,
                                        image_format: Format::Bgra8Srgb,
                                        image_color_space: ColorSpace::SrgbNonlinear,
                                        image_extent: Extent2D { width: 100, height: 100 },
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
                                VkResult::handle(unsafe { vkCreateSwapchainKHR(device, &swapchain_info, ptr::null(), &mut swapchain) })?;
                                Ok(swapchain)
                        }
                }

                pub struct DescriptorTable(BTreeMap<DescriptorType, u32>);

                impl Default for DescriptorTable {
                        fn default() -> Self {
                                let table = [DescriptorType::StorageBuffer, DescriptorType::StorageImage].into_iter().map(|x| (x, 10000)).collect();
                                Self(table)
                        }
                }

                #[repr(C)]
                pub struct DescriptorSetAllocateInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        descriptor_pool: DescriptorPool,
                        descriptor_set_count: u32,
                        p_set_layouts: *const DescriptorSetLayout,
                }

                #[derive(Clone, Copy)]
                pub struct DescriptorSet(*const c_void);
                impl Default for DescriptorSet {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }
                pub struct DescriptorSets(Vec<DescriptorSet>);

                impl DescriptorSets {
                        fn allocate(device: Device, descriptor_pool: DescriptorPool, descriptor_set_layout: DescriptorSetLayout) -> Result<Self, Error> {
                                let set_layouts = vec![descriptor_set_layout; 4];
                                let alloc_info = DescriptorSetAllocateInfo {
                                        s_type: StructureType::DescriptorSetAllocateInfo,
                                        p_next: ptr::null(),
                                        descriptor_pool,
                                        descriptor_set_count: 4,
                                        p_set_layouts: set_layouts.as_ptr(),
                                };

                                let mut descriptor_sets = vec![DescriptorSet::default(); 4];
                                VkResult::handle(unsafe { vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.as_mut_ptr()) })?;

                                Ok(Self(descriptor_sets))
                        }
                }

                #[derive(Clone, Copy)]
                pub struct DescriptorPool(*const c_void);

                impl Default for DescriptorPool {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }
                impl DescriptorPool {
                        fn bindless(device: Device, DescriptorTable(table): &DescriptorTable) -> Result<Self, Error> {
                                let pool_sizes = table.iter().map(|(&ty, &descriptor_count)| DescriptorPoolSize {
                                        ty,
                                        descriptor_count,
                                }).cycle().take(4 * table.len()).collect::<Vec<_>>();
                                let flags = [
                                        DescriptorPoolCreateFlagBits::UpdateAfterBind,
                                ].into_iter().fold(0, |accum, x| accum | x as u32);
                                let descriptor_pool_info = DescriptorPoolCreateInfo {
                                        s_type: StructureType::DescriptorPoolCreateInfo,
                                        p_next: ptr::null(),
                                        flags,
                                        max_sets: 4,
                                        pool_size_count: pool_sizes.len() as u32,
                                        p_pool_sizes: pool_sizes.as_ptr(),
                                };
                                let mut descriptor_pool = Self::default();
                                VkResult::handle(unsafe { vkCreateDescriptorPool(device, &descriptor_pool_info, ptr::null(), &mut descriptor_pool) })?;
                                Ok(descriptor_pool)
                        }
                }

                #[repr(C)]
                pub struct DescriptorPoolCreateInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        flags: DescriptorPoolCreateFlags,
                        max_sets: u32,
                        pool_size_count: u32,
                        p_pool_sizes: *const DescriptorPoolSize,
                }


                pub type DescriptorPoolCreateFlags = u32;


                #[repr(C)]
                #[derive(Clone, Copy, Debug, PartialEq, Eq)]
                pub enum DescriptorPoolCreateFlagBits {
                        FreeDescriptorSet = 0x00000001,
                        UpdateAfterBind = 0x00000002,
                        HostOnlyVALVE = 0x00000004,
                        AllowOverallocationNV = 0x00000008,
                }

                #[repr(C)]
                pub struct DescriptorPoolSize {
                        ty: DescriptorType,
                        descriptor_count: u32,
                }

                #[repr(C)]
                #[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
                pub enum DescriptorType {
                        Sampler = 0,
                        CombinedImageSampler = 1,
                        SampledImage = 2,
                        StorageImage = 3,
                        UniformTexelBuffer = 4,
                        StorageTexelBuffer = 5,
                        UniformBuffer = 6,
                        StorageBuffer = 7,
                        UniformBufferDynamic = 8,
                        StorageBufferDynamic = 9,
                        InputAttachment = 10,
                }

                #[repr(C)]
                pub struct DescriptorSetLayoutCreateInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        flags: DescriptorSetLayoutCreateFlags,
                        binding_count: u32,
                        p_bindings: *const DescriptorSetLayoutBinding,
                }

                #[repr(C)]
                pub struct DescriptorSetLayoutBindingFlagsCreateInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        binding_count: u32,
                        flags: *const DescriptorBindingFlags,
                }
                pub type DescriptorBindingFlags = u32;
                pub type DescriptorSetLayoutCreateFlags = u32;
                #[repr(C)]
                pub enum DescriptorBindingFlagBits {
                        UpdateAfterBind = 0x00000001,
                        UpdateUnusedWhilePending = 0x00000002,
                        PartiallyBound = 0x00000004,
                        VariableDescriptorCount = 0x00000008,
                }

                #[repr(C)]
                pub struct DescriptorSetLayoutBinding {
                        binding: u32,
                        descriptor_type: DescriptorType,
                        descriptor_count: u32,
                        stage_flags: ShaderStageFlags,
                        p_immutable_samplers: *const c_void,
                }

                #[repr(u32)]
                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                pub enum DescriptorSetLayoutCreateFlagBits {
                        UpdateAfterBindPool = 0x00000002,
                        PushDescriptorKhr = 0x00000001,
                        DescriptorBufferExt = 0x00000010,
                        EmbeddedImmutableSamplersExt = 0x00000020,
                        IndirectBindableNv = 0x00000080,
                        HostOnlyPoolExt = 0x00000004,
                        PerStageNv = 0x00000040,
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct DescriptorSetLayout(*const c_void);

                impl Default for DescriptorSetLayout {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }
                impl DescriptorSetLayout {
                        fn bindless(device: Device, DescriptorTable(table): &DescriptorTable) -> Result<Self, Error> {
                                let bindings = table.iter().enumerate().map(|(i, (descriptor_type, descriptor_count))| DescriptorSetLayoutBinding {
                                        binding: i as u32,
                                        descriptor_type: *descriptor_type,
                                        descriptor_count: *descriptor_count,
                                        stage_flags: ShaderStageFlagBits::All as u32,
                                        p_immutable_samplers: ptr::null(),
                                }).collect::<Vec<_>>();

                                let bind_flag = [
                                        DescriptorBindingFlagBits::PartiallyBound, DescriptorBindingFlagBits::UpdateAfterBind
                                ].into_iter().fold(0, |accum, x| accum | x as u32);
                                let bind_flags = vec![bind_flag; bindings.len()];

                                let binding_flags = DescriptorSetLayoutBindingFlagsCreateInfo {
                                        s_type: StructureType::DescriptorSetLayoutBindingFlagsCreateInfo,
                                        p_next: ptr::null(),
                                        binding_count: bindings.len() as u32,
                                        flags: bind_flags.as_ptr(),
                                };

                                let flags = [
                                        DescriptorSetLayoutCreateFlagBits::UpdateAfterBindPool,
                                ].into_iter().fold(0, |accum, x| accum | x as u32);

                                let descriptor_set_layout_info = DescriptorSetLayoutCreateInfo {
                                        s_type: StructureType::DescriptorSetLayoutCreateInfo,
                                        p_next: &binding_flags as *const _ as *const _,
                                        flags,
                                        binding_count: bindings.len() as u32,
                                        p_bindings: bindings.as_ptr(),
                                };

                                let mut descriptor_set_layout = Self::default();
                                VkResult::handle(unsafe { vkCreateDescriptorSetLayout(device, &descriptor_set_layout_info, ptr::null(), &mut descriptor_set_layout) })?;
                                Ok(descriptor_set_layout)
                        }
                }

                pub struct ShaderSource(Vec<u32>);

                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, Eq)]
                pub enum ShaderStageFlagBits {
                        Vertex = 0x00000001,
                        TessellationControl = 0x00000002,
                        TessellationEvaluation = 0x00000004,
                        Geometry = 0x00000008,
                        Fragment = 0x00000010,
                        Compute = 0x00000020,
                        AllGraphics = 0x0000001F,
                        All = 0x7FFFFFFF,
                }

                pub type ShaderCreateFlags = u32;
                pub type ShaderStageFlags = u32;

                #[repr(C)]
                #[derive(Debug, Copy, Clone)]
                pub enum ShaderCreateFlagBits {
                        LinkStage = 0x00000001,
                        AllowVaryingSubgroupSize = 0x00000002,
                        RequireFullSubgroups = 0x00000004,
                        NoTaskShader = 0x00000008,
                        DispatchBase = 0x00000010,
                        FragmentShadingRateAttachment = 0x00000020,
                        FragmentDensityMapAttachment = 0x00000040,
                }

                #[repr(C)]
                #[derive(Debug, Copy, Clone)]
                pub enum ShaderCodeType {
                        Binary = 0,
                        Spirv = 1,
                }

                #[repr(C)]
                pub struct ShaderCreateInfo {
                        pub s_type: StructureType,
                        pub p_next: *const c_void,
                        pub flags: ShaderCreateFlags,
                        pub stage: ShaderStageFlags,
                        pub next_stage: ShaderStageFlags,
                        pub code_type: ShaderCodeType,
                        pub code_size: usize,
                        pub p_code: *const c_void,
                        pub p_name: *const c_char,
                        pub set_layout_count: u32,
                        pub p_set_layouts: *const DescriptorSetLayout,
                        pub push_constant_range_count: u32,
                        pub p_push_constant_ranges: *const PushConstantRange,
                        pub p_specialization_info: *const SpecializationInfo,
                }

                #[derive(Clone, Copy)]
                pub struct PushConstantRange {
                        stage: ShaderStageFlags,
                        offset: u32,
                        size: u32,
                }

                #[repr(C)]
                pub struct SpecializationInfo {
                        pub map_entry_count: u32,
                        pub p_map_entries: *const SpecializationMapEntry,
                        pub data_size: usize,
                        pub p_data: *const c_void,
                }

                #[repr(C)]
                pub struct SpecializationMapEntry {
                        pub constant_id: u32,
                        pub offset: u32,
                        pub size: usize,
                }

                pub struct ShaderSpec {
                        flags: ShaderCreateFlags,
                        source: ShaderSource,
                        stage: ShaderStageFlags,
                        next_stage: ShaderStageFlags,
                        push_constant_range: PushConstantRange,
                        entry: String,
                }

                #[derive(Clone, Copy)]
                pub struct Shader(*const c_void);

                impl Default for Shader {
                        fn default() -> Self {
                                Self(ptr::null())
                        }
                }

                impl Shader {
                        fn compile(device: Device, set_layouts: &[DescriptorSetLayout], spec: Vec<ShaderSpec>) -> Result<Vec<Self>, Error> {
                                let source = spec.iter().map(|spec| spec.source.0.clone()).collect::<Vec<_>>();

                                let name = spec.iter().map(|spec| spec.entry.as_ref()).map(CString::from_str).map(|x| x.unwrap()).collect::<Vec<_>>();

                                let info = (0..spec.len()).map(|i| ShaderCreateInfo {
                                        s_type: StructureType::ShaderCreateInfo,
                                        p_next: ptr::null(),
                                        flags: 0,
                                        stage: spec[i].stage as u32,
                                        next_stage: spec[i].next_stage as u32,
                                        code_type: ShaderCodeType::Spirv,
                                        code_size: source[i].len(),
                                        p_code: source[i].as_ptr() as *const _,
                                        p_name: name[i].as_ptr(),
                                        set_layout_count: set_layouts.len() as u32,
                                        p_set_layouts: set_layouts.as_ptr(),
                                        push_constant_range_count: 1,
                                        p_push_constant_ranges: &spec[i].push_constant_range,
                                        p_specialization_info: &SpecializationInfo {
                                                map_entry_count: 0,
                                                p_map_entries: ptr::null(),
                                                data_size: 0,
                                                p_data: ptr::null(),
                                        },
                                }).collect::<Vec<_>>();

                                let mut shaders = vec![Self::default(); info.len()];

                                VkResult::handle(unsafe { vkCreateShadersEXT(device, info.len() as u32, info.as_ptr(), ptr::null(), shaders.as_mut_ptr()) })?;

                                Ok(shaders)
                        }
                }

                #[derive(Debug)]
                pub enum Error {
                        Unknown
                }

                pub struct ContextInfo {
                        app_name: String,
                        app_version: Version,
                }

                pub struct Context {
                        instance: Instance,
                        physical_device: PhysicalDevice,
                        device: Device,
                        surface: Surface,
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

                #[repr(u32)]
                pub enum StructureType {
                        ApplicationInfo = 0,
                        InstanceCreateInfo = 1,
                        DeviceQueueInfo = 2,
                        DeviceInfo = 3,
                        DescriptorSetLayoutCreateInfo = 32,
                        DescriptorPoolCreateInfo = 33,
                        DescriptorSetAllocateInfo = 34,
                        PhysicalDeviceFeatures2 = 1000059000,
                        PhysicalDeviceDescriptorIndexingFeatures = 1000161001,
                        SwapchainCreateInfo = 1000001000,
                        XlibSurfaceCreateInfo = 1000004000,
                        ShaderCreateInfo = 1000482002,
                        DescriptorSetLayoutBindingFlagsCreateInfo = 1000161000,
                }

                #[repr(C)]
                struct ApplicationInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        app_name: *const c_char,
                        app_version: u32,
                        engine_name: *const c_char,
                        engine_version: u32,
                        api_version: u32,
                }

                #[repr(C)]
                struct InstanceCreateInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        flags: u64,
                        app_info: *const ApplicationInfo,
                        enabled_layer_count: u32,
                        enabled_layer: *const *const c_char,
                        enabled_ext_count: u32,
                        enabled_ext: *const *const c_char,
                }

                pub type DeviceSize = u64;
                pub type Bool32 = u32;
                pub type SampleCountFlags = u32;

                #[repr(C)]
                pub struct PhysicalDeviceDescriptorIndexingFeatures {
                        s_type: StructureType,
                        p_next: *const c_void,
                        shader_input_attachment_array_dynamic_indexing: Bool32,
                        shader_uniform_texel_buffer_array_dynamic_indexing: Bool32,
                        shader_storage_texel_buffer_array_dynamic_indexing: Bool32,
                        shader_uniform_buffer_array_non_uniform_indexing: Bool32,
                        shader_sampled_image_array_non_uniform_indexing: Bool32,
                        shader_storage_buffer_array_non_uniform_indexing: Bool32,
                        shader_storage_image_array_non_uniform_indexing: Bool32,
                        shader_input_attachment_array_non_uniform_indexing: Bool32,
                        shader_uniform_texel_buffer_array_non_uniform_indexing: Bool32,
                        shader_storage_texel_buffer_array_non_uniform_indexing: Bool32,
                        descriptor_binding_uniform_buffer_update_after_bind: Bool32,
                        descriptor_binding_sampled_image_update_after_bind: Bool32,
                        descriptor_binding_storage_image_update_after_bind: Bool32,
                        descriptor_binding_storage_buffer_update_after_bind: Bool32,
                        descriptor_binding_uniform_texel_buffer_update_after_bind: Bool32,
                        descriptor_binding_storage_texel_buffer_update_after_bind: Bool32,
                        descriptor_binding_update_unused_while_pending: Bool32,
                        descriptor_binding_partially_bound: Bool32,
                        descriptor_binding_variable_descriptor_count: Bool32,
                        runtime_descriptor_array: Bool32,
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct PhysicalDeviceLimits {
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
                        pub buffer_image_granularity: DeviceSize,
                        pub sparse_address_space_size: DeviceSize,
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
                        pub min_texel_buffer_offset_alignment: DeviceSize,
                        pub min_uniform_buffer_offset_alignment: DeviceSize,
                        pub min_storage_buffer_offset_alignment: DeviceSize,
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
                        pub framebuffer_color_sample_counts: SampleCountFlags,
                        pub framebuffer_depth_sample_counts: SampleCountFlags,
                        pub framebuffer_stencil_sample_counts: SampleCountFlags,
                        pub framebuffer_no_attachments_sample_counts: SampleCountFlags,
                        pub max_color_attachments: u32,
                        pub sampled_image_color_sample_counts: SampleCountFlags,
                        pub sampled_image_integer_sample_counts: SampleCountFlags,
                        pub sampled_image_depth_sample_counts: SampleCountFlags,
                        pub sampled_image_stencil_sample_counts: SampleCountFlags,
                        pub storage_image_sample_counts: SampleCountFlags,
                        pub max_sample_mask_words: u32,
                        pub timestamp_compute_and_graphics: Bool32,
                        pub timestamp_period: f32,
                        pub max_clip_distances: u32,
                        pub max_cull_distances: u32,
                        pub max_combined_clip_and_cull_distances: u32,
                        pub discrete_queue_priorities: u32,
                        pub point_size_range: [f32; 2],
                        pub line_width_range: [f32; 2],
                        pub point_size_granularity: f32,
                        pub line_width_granularity: f32,
                        pub strict_lines: Bool32,
                        pub standard_sample_locations: Bool32,
                        pub optimal_buffer_copy_offset_alignment: DeviceSize,
                        pub optimal_buffer_copy_row_pitch_alignment: DeviceSize,
                        pub non_coherent_atom_size: DeviceSize,
                }

                #[repr(C)]
                #[derive(Debug, Copy, Clone, PartialEq, Eq)]
                pub enum PhysicalDeviceType {
                        Other = 0,
                        IntegratedGpu = 1,
                        DiscreteGpu = 2,
                        VirtualGpu = 3,
                        Cpu = 4,
                }

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct PhysicalDeviceSparseProperties {
                        pub residency_standard_2d_block_shape: Bool32,
                        pub residency_standard_2d_multisample_block_shape: Bool32,
                        pub residency_standard_3d_block_shape: Bool32,
                        pub residency_aligned_mip_size: Bool32,
                        pub residency_non_resident_strict: Bool32,
                }

                pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
                pub const VK_UUID_SIZE: usize = 16;

                #[repr(C)]
                #[derive(Clone, Copy)]
                pub struct PhysicalDeviceProperties {
                        pub api_version: u32,
                        pub driver_version: u32,
                        pub vendor_id: u32,
                        pub device_id: u32,
                        pub device_type: PhysicalDeviceType,
                        pub device_name: [u8; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
                        pub pipeline_cache_uuid: [u8; VK_UUID_SIZE],
                        pub limits: PhysicalDeviceLimits,
                        pub sparse_properties: PhysicalDeviceSparseProperties,
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

                #[repr(C)]
                pub struct DeviceQueueCreateInfo {
                        s_type: StructureType,
                        p_next: *const c_void,
                        flags: DeviceQueueCreateFlags,
                        queue_family_index: QueueFamilyIndex,
                        queue_count: u32,
                        queue_priorities: *const f32,
                }


                #[repr(C)]
                #[derive(Default)]
                pub struct PhysicalDeviceFeatures {
                        pub robust_buffer_access: Bool32,
                        pub full_draw_index_uint32: Bool32,
                        pub image_cube_array: Bool32,
                        pub independent_blend: Bool32,
                        pub geometry_shader: Bool32,
                        pub tessellation_shader: Bool32,
                        pub sample_rate_shading: Bool32,
                        pub dual_src_blend: Bool32,
                        pub logic_op: Bool32,
                        pub multi_draw_indirect: Bool32,
                        pub draw_indirect_first_instance: Bool32,
                        pub depth_clamp: Bool32,
                        pub depth_bias_clamp: Bool32,
                        pub fill_mode_non_solid: Bool32,
                        pub depth_bounds: Bool32,
                        pub wide_lines: Bool32,
                        pub large_points: Bool32,
                        pub alpha_to_one: Bool32,
                        pub multi_viewport: Bool32,
                        pub sampler_anisotropy: Bool32,
                        pub texture_compression_etc2: Bool32,
                        pub texture_compression_astc_ldr: Bool32,
                        pub texture_compression_bc: Bool32,
                        pub occlusion_query_precise: Bool32,
                        pub pipeline_statistics_query: Bool32,
                        pub vertex_pipeline_stores_and_atomics: Bool32,
                        pub fragment_stores_and_atomics: Bool32,
                        pub shader_tessellation_and_geometry_point_size: Bool32,
                        pub shader_image_gather_extended: Bool32,
                        pub shader_storage_image_extended_formats: Bool32,
                        pub shader_storage_image_multisample: Bool32,
                        pub shader_storage_image_read_without_format: Bool32,
                        pub shader_storage_image_write_without_format: Bool32,
                        pub shader_uniform_buffer_array_dynamic_indexing: Bool32,
                        pub shader_sampled_image_array_dynamic_indexing: Bool32,
                        pub shader_storage_buffer_array_dynamic_indexing: Bool32,
                        pub shader_storage_image_array_dynamic_indexing: Bool32,
                        pub shader_clip_distance: Bool32,
                        pub shader_cull_distance: Bool32,
                        pub shader_float64: Bool32,
                        pub shader_int64: Bool32,
                        pub shader_int16: Bool32,
                        pub shader_resource_residency: Bool32,
                        pub shader_resource_min_lod: Bool32,
                        pub sparse_binding: Bool32,
                        pub sparse_residency_buffer: Bool32,
                        pub sparse_residency_image2_d: Bool32,
                        pub sparse_residency_image3_d: Bool32,
                        pub sparse_residency2_samples: Bool32,
                        pub sparse_residency4_samples: Bool32,
                        pub sparse_residency8_samples: Bool32,
                        pub sparse_residency16_samples: Bool32,
                        pub sparse_residency_aliased: Bool32,
                        pub variable_multisample_rate: Bool32,
                        pub inherited_queries: Bool32,
                }

                pub type QueueFlags = u32;

                #[repr(C)]
                #[derive(Debug, Clone, Copy)]
                pub struct Extent3D {
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
                pub struct QueueFamilyProperties {
                        pub queue_flags: QueueFlags,
                        pub queue_count: u32,
                        pub timestamp_valid_bits: u32,
                        pub min_image_transfer_granularity: Extent3D,
                }

                pub type DeviceQueueCreateFlags = u32;

                #[repr(C)]
                pub struct XlibSurfaceCreateInfo {
                        pub s_type: StructureType,
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
                        OutOfPoolMemory = -1000069000,
                }

                impl VkResult {
                        pub(crate) fn handle(r: VkResult) -> Result<(), Error> {
                                match r {
                                        VkResult::Success => Ok(()),
                                        e => {
                                                panic!("{e:?}");
                                                Err(Error::Unknown)
                                        }
                                }
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

                #[repr(C)]
                pub enum Format {
                        Undefined = 0,
                        Bgra8Srgb = 50,
                        // ... other variants ...
                }

                #[repr(C)]
                pub enum ColorSpace {
                        SrgbNonlinear = 0,
                        // ... other variants ...
                }

                #[repr(C)]
                pub struct Extent2D {
                        pub width: u32,
                        pub height: u32,
                }

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

                #[repr(C)]
                pub enum SharingMode {
                        Exclusive = 0,
                        Concurrent = 1,
                }

                #[repr(C)]
                pub enum SurfaceTransformFlagBits {
                        IdentityBit = 0x00000001,
                        // ... other variants ...
                }

                #[repr(C)]
                pub enum CompositeAlphaFlagBits {
                        OpaqueBit = 0x00000001,
                        // ... other variants ...
                }

                #[repr(C)]
                pub enum PresentMode {
                        Immediate = 0,
                        Mailbox = 1,
                        Fifo = 2,
                        // ... other variants ...
                }

                extern "C" {
                        fn vkCreateInstance(info: *const InstanceCreateInfo, alloc: *const c_void, out: *mut Instance) -> VkResult;
                        fn vkEnumeratePhysicalDevices(instance: Instance, count: *mut u32, physical_devices: *mut PhysicalDevice) -> VkResult;
                        fn vkGetPhysicalDeviceProperties(physical_device: PhysicalDevice, property: *mut PhysicalDeviceProperties);
                        fn vkCreateDevice(physical_device: PhysicalDevice, info: *const DeviceCreateInfo, alloc: *const c_void, out: *mut Device) -> VkResult;
                        fn vkGetPhysicalDeviceQueueFamilyProperties(physical_device: PhysicalDevice, count: *mut u32, physical_devices: *mut QueueFamilyProperties) -> VkResult;
                        fn vkCreateXlibSurfaceKHR(instance: Instance, info: *const XlibSurfaceCreateInfo, alloc: *const c_void, surface: *mut Surface) -> VkResult;
                        fn vkCreateSwapchainKHR(device: Device, info: *const SwapchainCreateInfo, alloc: *const c_void, swapchain: *mut Swapchain) -> VkResult;
                        fn vkCreateShadersEXT(device: Device, count: u32, infos: *const ShaderCreateInfo, alloc: *const c_void, shaders: *mut Shader) -> VkResult;
                        fn vkCreateDescriptorPool(device: Device, info: *const DescriptorPoolCreateInfo, alloc: *const c_void, descriptor_pool: *mut DescriptorPool) -> VkResult;
                        fn vkCreateDescriptorSetLayout(device: Device, info: *const DescriptorSetLayoutCreateInfo, alloc: *const c_void, descriptor_pool: *mut DescriptorSetLayout) -> VkResult;
                        fn vkAllocateDescriptorSets(device: Device, info: *const DescriptorSetAllocateInfo, descriptor_sets: *mut DescriptorSet) -> VkResult;
                }
        }

        #[cfg(target_os = "linux")]
        mod x11 {
                use core::ffi::c_void;
                use core::ptr;

                use libc::c_char;

                use crate::gfx::{Surface, SurfaceHandle};

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

                        fn handle(&self) -> SurfaceHandle {
                                SurfaceHandle::Linux { display: self.display, window: self.window }
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
        }


        pub fn make_platform_surface() -> Box<dyn Surface> {
                #[cfg(target_os = "linux")]
                Box::new(x11::Window::open())
        }

        pub fn make_platform_renderer() -> Box<dyn Renderer> {
                let surface = make_platform_surface();
                #[cfg(target_os = "linux")]
                Box::new(vk::Renderer::create(&*surface))
        }

        pub enum SurfaceHandle {
                Linux { window: *const c_void, display: *const c_void },
        }

        pub trait Surface {
                fn open() -> Self
                where
                        Self: Sized;
                fn update(&self);
                fn handle(&self) -> SurfaceHandle;
        }

        pub trait Renderer {
                fn create(surface: &dyn Surface) -> Self
                where
                        Self: Sized;
                fn render(&mut self);
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
