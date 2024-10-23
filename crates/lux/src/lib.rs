#![feature(allocator_api, try_blocks, iterator_try_collect, generic_const_exprs)]
#![feature(iter_chain)]
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

    use crate::math::{Quaternion, Vector};
    use crate::prelude::*;

    #[cfg(target_os = "linux")]
    pub mod vk {
        use core::{iter, mem, ptr};
        use core::ffi::{c_void, CStr};
        use core::ops::Add;
        use core::ptr::slice_from_raw_parts;
        use core::str::FromStr;

        use libc::c_char;

        use crate::env::{Console, Level};
        use crate::gfx::{Camera, Renderer as GfxRenderer, Surface as GfxSurface, SurfaceHandle};
        use crate::gfx::vk::cmd::BufferBarrier;
        use crate::prelude::*;

        pub struct Renderer {
            instance: Instance,
            physical_device: PhysicalDevice,
            device: Device,
            queue: Queue,
            qfi: QueueFamilyIndex,
            surface: Surface,
            swapchain: Swapchain,
            swapchain_images: Vec<Image>,
            swapchain_image_views: Vec<ImageView>,
            descriptor_table: DescriptorTable,
            descriptor_pool: DescriptorPool,
            descriptor_set_layout: DescriptorSetLayout,
            descriptor_sets: Vec<DescriptorSet>,
            shaders: Vec<Shader>,
            command_pool: CommandPool,
            command_buffers: Vec<CommandBuffer>,
            in_flight_fence: Vec<Fence>,
            image_avail_semaphore: Vec<Semaphore>,
            render_finish_semaphore: Vec<Semaphore>,
            pipeline_layout: PipelineLayout,
            staging_buffer: StagingBuffer,
            frame: usize,

            global_buffer: Buffer,

            camera: Camera,
        }

        impl GfxRenderer for Renderer {
            fn create(surface: &dyn GfxSurface) -> Self
            where
                Self: Sized,
            {
                let instance = Instance::new().unwrap();
                let physical_device = PhysicalDevice::acquire_best(instance).unwrap();
                let surface = Surface::open(instance, surface).unwrap();
                let (device, qfi) = Device::new(physical_device, surface).unwrap();
                let queue = qfi.queue(device);
                let resolution = resolution(physical_device, surface);
                let (swapchain, swapchain_images) =
                    Swapchain::new(device, qfi, surface, None, resolution).unwrap();
                let swapchain_image_views = swapchain_images
                    .iter()
                    .copied()
                    .map(|image| {
                        ImageView::create(device, image, ImageViewType::Type2d, Format::Bgra8Unorm)
                    })
                    .try_collect::<Vec<_>>()
                    .unwrap();
                let descriptor_table = DescriptorTable::default();
                let descriptor_pool = DescriptorPool::bindless(device, &descriptor_table).unwrap();
                let descriptor_set_layout =
                    DescriptorSetLayout::bindless(device, &descriptor_table).unwrap();
                let descriptor_sets =
                    DescriptorSet::allocate(device, descriptor_pool, descriptor_set_layout)
                        .unwrap();
                let vertex = include_bytes!("../../../assets/shader/vertex.spirv").to_vec();
                let fragment = include_bytes!("../../../assets/shader/fragment.spirv").to_vec();
                let shader_specs = vec![
                    ShaderSpec {
                        flags: 0,
                        source: ShaderSource(vertex),
                        stage: ShaderStageFlagBits::Vertex as u32,
                        next_stage: ShaderStageFlagBits::Fragment as u32,
                        push_constant_range: PushConstantRange {
                            stage: ShaderStageFlagBits::All as u32,
                            offset: 0,
                            size: 64,
                        },
                        entry: "main".to_string(),
                    },
                    ShaderSpec {
                        flags: 0,
                        source: ShaderSource(fragment),
                        stage: ShaderStageFlagBits::Fragment as u32,
                        next_stage: 0,
                        push_constant_range: PushConstantRange {
                            stage: ShaderStageFlagBits::All as u32,
                            offset: 0,
                            size: 64,
                        },
                        entry: "main".to_string(),
                    },
                ];
                let shaders =
                    Shader::compile(device, &[descriptor_set_layout], shader_specs).unwrap();
                let command_pool = CommandPool::new(device, qfi).unwrap();
                let command_buffers = CommandBuffer::allocate(device, command_pool, 3).unwrap();
                let in_flight_fence = iter::repeat_with(|| Fence::create_signaled(device))
                    .take(3)
                    .try_collect::<Vec<_>>()
                    .unwrap();
                let image_avail_semaphore = iter::repeat_with(|| Semaphore::create(device))
                    .take(3)
                    .try_collect::<Vec<_>>()
                    .unwrap();
                let render_finish_semaphore = iter::repeat_with(|| Semaphore::create(device))
                    .take(3)
                    .try_collect::<Vec<_>>()
                    .unwrap();

                let global_buffer = Buffer::create(
                    device,
                    1024,
                    BufferUsageFlagBits::StorageBuffer as u32
                        | BufferUsageFlagBits::TransferDst as u32,
                );

                let memory = {
                    let req = global_buffer.requirements(device);
                    Memory::device_local(physical_device, device, req)
                        .bind_buffer(device, global_buffer)
                };

                let staging_buffer = StagingBuffer::create(physical_device, device, 1_000_000);

                let pipeline_layout = PipelineLayout::new(device, descriptor_set_layout);

                Self {
                    instance,
                    physical_device,
                    device,
                    qfi,
                    surface,
                    swapchain,
                    swapchain_images,
                    swapchain_image_views,
                    descriptor_table,
                    descriptor_pool,
                    descriptor_set_layout,
                    descriptor_sets,
                    shaders,
                    command_pool,
                    command_buffers,
                    in_flight_fence,
                    image_avail_semaphore,
                    render_finish_semaphore,
                    queue,
                    global_buffer,
                    pipeline_layout,
                    staging_buffer,
                    frame: 0,
                    camera: Camera::default(),
                }
            }

            fn render(&mut self) {
                if self.in_flight_fence[self.frame].wait(self.device) {
                    return;
                }

                self.in_flight_fence[self.frame].reset(self.device);
                let (width, height) = resolution(self.physical_device, self.surface);

                self.staging_buffer.write(
                    self.global_buffer,
                    0,
                    [self.frame as f32 / 3.0, ((self.frame + 1) % 3) as f32 / 3.0, ((self.frame + 2) % 3) as f32 / 3.0],
                );
                self.staging_buffer.host_transfer(self.device);

                let Ok(idx) = self
                    .swapchain
                    .next_image(self.device, self.image_avail_semaphore[self.frame])
                else {
                    let (swapchain, swapchain_images) = Swapchain::new(
                        self.device,
                        self.qfi,
                        self.surface,
                        Some(self.swapchain),
                        (width, height),
                    )
                    .unwrap();
                    self.swapchain = swapchain;
                    self.swapchain_images = swapchain_images;
                    self.swapchain_image_views = self
                        .swapchain_images
                        .iter()
                        .copied()
                        .map(|image| {
                            ImageView::create(
                                self.device,
                                image,
                                ImageViewType::Type2d,
                                Format::Bgra8Unorm,
                            )
                        })
                        .try_collect::<Vec<_>>()
                        .unwrap();
                    return;
                };

                let buf = self.command_buffers[idx as usize];
                cmd::begin(buf);

                cmd::pipeline_barrier_buffer(
                    buf,
                    PipelineStageFlagBits::TopOfPipe as u32,
                    PipelineStageFlagBits::Transfer as u32,
                    &[

                        BufferBarrier {
                        src_access: 0,
                        dst_access: AccessFlagBits::TransferWrite as u32,
                        buffer: self.global_buffer,
                        offset: 0,
                        size: 64,
                    }],
                );

                self.staging_buffer.device_copy(buf);

                cmd::pipeline_barrier_buffer(
                    buf,
                    PipelineStageFlagBits::Transfer as u32,
                    PipelineStageFlagBits::VertexShader as u32,
                    &[BufferBarrier {
                        src_access: AccessFlagBits::TransferWrite as u32,
                        dst_access: AccessFlagBits::ShaderRead as u32,
                        buffer: self.global_buffer,
                        offset: 0,
                        size: 64,
                    }],
                );

                cmd::pipeline_barrier_image(
                    buf,
                    0,
                    PipelineStageFlagBits::TopOfPipe as u32,
                    AccessFlagBits::ColorAttachmentWrite as u32,
                    PipelineStageFlagBits::ColorAttachmentOutput as u32,
                    vec![(
                        self.swapchain_images[idx as usize],
                        ImageLayout::Undefined,
                        ImageLayout::ColorAttachmentOptimal,
                    )],
                );
                cmd::begin_render(
                    buf,
                    Render {
                        width,
                        height,
                        color: vec![RenderAttachment {
                            image_view: self.swapchain_image_views[idx as usize],
                            image_layout: ImageLayout::ColorAttachmentOptimal,
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: ClearValue {
                                color: ClearColorValue {
                                    float32: [1.0, 0.0, 1.0, 1.0],
                                },
                            },
                        }],
                        depth: None,
                        stencil: None,
                    },
                );

                cmd::push_constant(
                    buf,
                    self.pipeline_layout,
                    PushConstant {
                        offset: 0,
                        data: Push {
                            global_buffer: self.global_buffer.device_address(self.device),
                        },
                    },
                );
                cmd::bind_shaders(
                    self.device,
                    buf,
                    vec![
                        ShaderStageFlagBits::Vertex as u32,
                        ShaderStageFlagBits::Fragment as u32,
                    ],
                    self.shaders.clone(),
                );
                cmd::draw_settings(self.device, buf, width, height);
                cmd::draw(buf);
                cmd::end_render(buf);
                cmd::pipeline_barrier_image(
                    buf,
                    AccessFlagBits::ColorAttachmentWrite as u32,
                    PipelineStageFlagBits::ColorAttachmentOutput as u32,
                    0,
                    PipelineStageFlagBits::BottomOfPipe as u32,
                    vec![(
                        self.swapchain_images[idx as usize],
                        ImageLayout::ColorAttachmentOptimal,
                        ImageLayout::Present,
                    )],
                );

                cmd::end(buf);

                self.queue.submit(
                    self.image_avail_semaphore[self.frame],
                    self.render_finish_semaphore[self.frame],
                    self.in_flight_fence[self.frame],
                    buf,
                );
                self.queue.present(
                    self.render_finish_semaphore[self.frame],
                    self.swapchain,
                    idx,
                );

                self.frame = (self.frame + 1) % 3;
            }

            fn camera(&mut self) -> &mut Camera {
                &mut self.camera
            }
        }

        pub struct StagedWrite {
            buffer: Buffer,
            from: DeviceSize,
            to: DeviceSize,
            size: DeviceSize,
        }

        pub struct StagingBuffer {
            buffer: Buffer,
            memory: Memory,
            writes: Vec<StagedWrite>,
            waiting: Vec<u8>,
            size: u64,
            cursor: u64,
            last_write: u64,
        }

        impl StagingBuffer {
            fn create(physical_device: PhysicalDevice, device: Device, size: u64) -> Self {
                let buffer = Buffer::create(device, size, BufferUsageFlagBits::TransferSrc as u32);

                let req = buffer.requirements(device);

                let memory = Memory::host(physical_device, device, req).bind_buffer(device, buffer);

                Self {
                    buffer,
                    memory,
                    size,
                    cursor: 0,
                    last_write: 0,
                    writes: vec![],
                    waiting: vec![],
                }
            }
            fn write<T: Copy>(&mut self, buffer: Buffer, offset: u64, data: T) {
                let data = unsafe {
                    slice_from_raw_parts(&data as *const _ as *const u8, mem::size_of::<T>())
                        .as_ref()
                        .unwrap()
                };
                self.writes.push(StagedWrite {
                    to: offset,
                    from: self.cursor,
                    size: data.len() as u64,
                    buffer,
                });
                self.waiting.extend(data);
                if self.cursor >= self.size {
                    self.cursor = 0;
                }
                self.cursor += data.len() as u64;
            }
            fn host_transfer(&mut self, device: Device) {
                let data = self
                    .memory
                    .map(device, self.last_write, self.waiting.len() as u64)
                    as *mut _;
                unsafe { ptr::copy(self.waiting.as_ptr(), data, self.waiting.len()) };
                self.memory.unmap(device);
                self.waiting.clear();
            }
            fn device_copy(&mut self, cmd: CommandBuffer) {
                cmd::pipeline_barrier_buffer(
                    cmd,
                    PipelineStageFlagBits::TopOfPipe as u32,
                    PipelineStageFlagBits::Transfer as u32,
                    &[

                        BufferBarrier {
                            src_access: 0,
                            dst_access: AccessFlagBits::TransferRead as u32,
                            buffer: self.buffer,
                            offset: self.last_write,
                            size: self.writes.iter().map(|x| x.size).sum(),
                        }],
                );
                for write in self.writes.drain(..) {
                    cmd::copy_buffer(
                        cmd,
                        self.buffer,
                        write.buffer,
                        write.from,
                        write.to,
                        write.size,
                    );
                }
                if self.cursor >= self.size {
                    self.cursor = 0;
                }
                self.last_write = self.cursor;
            }
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct PipelineLayout(*const c_void);

        impl Default for PipelineLayout {
            fn default() -> Self {
                Self(ptr::null())
            }
        }

        impl PipelineLayout {
            fn new(device: Device, descriptor_set_layout: DescriptorSetLayout) -> Self {
                let info = PipelineLayoutCreateInfo {
                    s_type: StructureType::PipelineLayoutCreateInfo,
                    p_next: ptr::null(),
                    flags: 0,
                    set_layout_count: 1,
                    p_set_layouts: &descriptor_set_layout,
                    push_constant_range_count: 1,
                    p_push_constant_ranges: &PushConstantRange {
                        stage: ShaderStageFlagBits::All as u32,
                        offset: 0,
                        size: 64,
                    },
                };

                let mut layout = Self::default();
                unsafe { vkCreatePipelineLayout(device, &info, ptr::null(), &mut layout) };
                layout
            }
        }

        pub type PipelineLayoutCreateFlags = u32;

        #[repr(C)]
        pub struct PipelineLayoutCreateInfo {
            s_type: StructureType,
            p_next: *const c_void,
            flags: PipelineLayoutCreateFlags,
            set_layout_count: u32,
            p_set_layouts: *const DescriptorSetLayout,
            push_constant_range_count: u32,
            p_push_constant_ranges: *const PushConstantRange,
        }

        #[derive(Clone, Copy)]
        pub struct DeviceAddress(u64);

        #[derive(Clone, Copy)]
        pub struct Push {
            global_buffer: DeviceAddress,
        }

        pub struct PushConstant<T: Copy> {
            offset: u32,
            data: T,
        }

        fn resolution(physical_device: PhysicalDevice, surface: Surface) -> (u32, u32) {
            let mut info = unsafe { mem::zeroed::<SurfaceCapabilitiesKHR>() };
            unsafe {
                vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &mut info)
            };
            (info.current_extent.width, info.current_extent.height)
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

                let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) =
                    c_str_array(&*ext_instance);

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
                VkResult::handle(unsafe {
                    vkCreateInstance(&instance_info, ptr::null(), &mut instance)
                })?;
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

                VkResult::handle(unsafe {
                    vkEnumeratePhysicalDevices(instance, &mut count, physical_devices.as_mut_ptr())
                })?;

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

                let (physical_device, physical_device_name) =
                    rank.pop_last().map(|(_, x)| x).unwrap();
                Console::log(
                    Level::Trace,
                    &format!("Using GPU: \"{physical_device_name}\""),
                );

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
            fn new(
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

                let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) =
                    c_str_array(&*ext_device);

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
                VkResult::handle(unsafe {
                    vkCreateXlibSurfaceKHR(instance, &surface_info, ptr::null(), &mut surface)
                })?;

                Ok(surface)
            }
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct QueueFamilyIndex(u32);

        impl QueueFamilyIndex {
            fn queue(self, device: Device) -> Queue {
                let mut queue = Queue::default();
                unsafe { vkGetDeviceQueue(device, self, 0, &mut queue) };
                queue
            }
            fn graphics(physical_device: PhysicalDevice, surface: Surface) -> Result<Self, Error> {
                let mut count = 8;
                let mut queue_family_properties =
                    vec![unsafe { mem::zeroed::<QueueFamilyProperties>() }; 8];
                unsafe {
                    vkGetPhysicalDeviceQueueFamilyProperties(
                        physical_device,
                        &mut count,
                        queue_family_properties.as_mut_ptr(),
                    )
                };

                let queue_family_index = queue_family_properties
                    .iter()
                    .enumerate()
                    .find(|(_index, properties)| {
                        let graphics = properties.queue_flags & QueueFlagBits::Graphics as u32 != 0;
                        let mut present = false as _;
                        unsafe {
                            vkGetPhysicalDeviceSurfaceSupportKHR(
                                physical_device,
                                *_index as u32,
                                surface,
                                &mut present,
                            )
                        };
                        let present = present == true as _;
                        graphics && present
                    })
                    .map(|(index, _properties)| index as u32)
                    .unwrap();

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
            fn next_image(self, device: Device, semaphore: Semaphore) -> Result<u32, Error> {
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
            fn new(
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

                unsafe {
                    vkGetSwapchainImagesKHR(device, swapchain, &mut count, images.as_mut_ptr())
                }

                images.truncate(count as usize);
                dbg!(&images);
                Ok((swapchain, images))
            }
        }

        #[repr(C)]
        pub struct CommandBufferAllocateInfo {
            s_type: StructureType,
            p_next: *const c_void,
            command_pool: CommandPool,
            level: CommandBufferLevel,
            command_buffer_count: u32,
        }

        #[repr(C)]
        pub enum CommandBufferLevel {
            Primary = 0,
            Secondary = 1,
        }

        pub type CommandPoolCreateFlags = u32;

        #[repr(C)]
        pub struct CommandPoolCreateInfo {
            pub s_type: StructureType,
            pub p_next: *const core::ffi::c_void,
            pub flags: CommandPoolCreateFlags,
            pub queue_family_index: QueueFamilyIndex,
        }

        pub type DependencyFlags = u32;

        #[repr(u32)]
        pub enum DependencyFlagBits {
            ByRegion = 0x00000001,
            DeviceGroup = 0x00000004,
            ViewLocal = 0x00000002,
            // Add other variants as per Vulkan spec
        }

        // Flag bits for AccessFlags
        #[repr(u32)]
        pub enum AccessFlagBits {
            IndirectCommandRead = 0x00000001,
            IndexRead = 0x00000002,
            VertexAttributeRead = 0x00000004,
            UniformRead = 0x00000008,
            InputAttachmentRead = 0x00000010,
            ShaderRead = 0x00000020,
            ShaderWrite = 0x00000040,
            ColorAttachmentRead = 0x00000080,
            ColorAttachmentWrite = 0x00000100,
            DepthStencilAttachmentRead = 0x00000200,
            DepthStencilAttachmentWrite = 0x00000400,
            TransferRead = 0x00000800,
            TransferWrite = 0x00001000,
            HostRead = 0x00002000,
            HostWrite = 0x00004000,
            MemoryRead = 0x00008000,
            MemoryWrite = 0x00010000,
            // Add other variants as per Vulkan spec
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

        pub type AccessFlags = u32;

        #[repr(C)]
        pub struct MemoryBarrier {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub src_access_mask: AccessFlags,
            pub dst_access_mask: AccessFlags,
        }

        #[repr(C)]
        pub struct BufferMemoryBarrier {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub src_access_mask: AccessFlags,
            pub dst_access_mask: AccessFlags,
            pub src_queue_family_index: u32,
            pub dst_queue_family_index: u32,
            pub buffer: Buffer,
            pub offset: DeviceSize,
            pub size: DeviceSize,
        }

        #[repr(C)]
        pub struct ImageMemoryBarrier {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub src_access_mask: AccessFlags,
            pub dst_access_mask: AccessFlags,
            pub old_layout: ImageLayout,
            pub new_layout: ImageLayout,
            pub src_queue_family_index: u32,
            pub dst_queue_family_index: u32,
            pub image: Image,
            pub subresource_range: ImageSubresourceRange,
        }

        #[derive(Clone, Copy)]
        pub struct CommandBuffer(*const c_void);

        impl Default for CommandBuffer {
            fn default() -> Self {
                Self(ptr::null())
            }
        }

        impl CommandBuffer {
            fn allocate(
                device: Device,
                command_pool: CommandPool,
                command_buffer_count: u32,
            ) -> Result<Vec<Self>, Error> {
                let alloc_info = CommandBufferAllocateInfo {
                    s_type: StructureType::CommandBufferAllocateInfo,
                    p_next: ptr::null(),
                    command_pool,
                    level: CommandBufferLevel::Primary,
                    command_buffer_count,
                };

                let mut command_buffers = vec![CommandBuffer::default(); 4];
                VkResult::handle(unsafe {
                    vkAllocateCommandBuffers(device, &alloc_info, command_buffers.as_mut_ptr())
                })?;

                Ok(command_buffers)
            }
        }

        #[repr(C)]
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub enum CommandPoolCreateFlagBits {
            Transient = 0x00000001,
            ResetCommandBuffer = 0x00000002,
            Protected = 0x00000004,
        }

        #[derive(Clone, Copy)]
        pub struct CommandPool(*const c_void);

        impl Default for CommandPool {
            fn default() -> Self {
                Self(ptr::null())
            }
        }
        impl CommandPool {
            fn new(device: Device, queue_family_index: QueueFamilyIndex) -> Result<Self, Error> {
                let command_pool_info = CommandPoolCreateInfo {
                    s_type: StructureType::CommandPoolCreateInfo,
                    p_next: ptr::null(),
                    flags: CommandPoolCreateFlagBits::ResetCommandBuffer as u32,
                    queue_family_index,
                };

                let mut command_pool = default();
                VkResult::handle(unsafe {
                    vkCreateCommandPool(device, &command_pool_info, ptr::null(), &mut command_pool)
                })?;

                Ok(command_pool)
            }
        }

        pub struct DescriptorTable(BTreeMap<DescriptorType, u32>);

        impl Default for DescriptorTable {
            fn default() -> Self {
                let table = [DescriptorType::StorageBuffer, DescriptorType::StorageImage]
                    .into_iter()
                    .map(|x| (x, 10000))
                    .collect();
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

        impl DescriptorSet {
            fn allocate(
                device: Device,
                descriptor_pool: DescriptorPool,
                descriptor_set_layout: DescriptorSetLayout,
            ) -> Result<Vec<Self>, Error> {
                let set_layouts = vec![descriptor_set_layout; 4];
                let alloc_info = DescriptorSetAllocateInfo {
                    s_type: StructureType::DescriptorSetAllocateInfo,
                    p_next: ptr::null(),
                    descriptor_pool,
                    descriptor_set_count: 4,
                    p_set_layouts: set_layouts.as_ptr(),
                };

                let mut descriptor_sets = vec![DescriptorSet::default(); 4];
                VkResult::handle(unsafe {
                    vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.as_mut_ptr())
                })?;

                Ok(descriptor_sets)
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
            fn bindless(
                device: Device,
                DescriptorTable(table): &DescriptorTable,
            ) -> Result<Self, Error> {
                let pool_sizes = table
                    .iter()
                    .map(|(&ty, &descriptor_count)| DescriptorPoolSize {
                        ty,
                        descriptor_count,
                    })
                    .cycle()
                    .take(4 * table.len())
                    .collect::<Vec<_>>();
                let flags = [DescriptorPoolCreateFlagBits::UpdateAfterBind]
                    .into_iter()
                    .fold(0, |accum, x| accum | x as u32);
                let descriptor_pool_info = DescriptorPoolCreateInfo {
                    s_type: StructureType::DescriptorPoolCreateInfo,
                    p_next: ptr::null(),
                    flags,
                    max_sets: 4,
                    pool_size_count: pool_sizes.len() as u32,
                    p_pool_sizes: pool_sizes.as_ptr(),
                };
                let mut descriptor_pool = Self::default();
                VkResult::handle(unsafe {
                    vkCreateDescriptorPool(
                        device,
                        &descriptor_pool_info,
                        ptr::null(),
                        &mut descriptor_pool,
                    )
                })?;
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
            fn bindless(
                device: Device,
                DescriptorTable(table): &DescriptorTable,
            ) -> Result<Self, Error> {
                let bindings = table
                    .iter()
                    .enumerate()
                    .map(
                        |(i, (descriptor_type, descriptor_count))| DescriptorSetLayoutBinding {
                            binding: i as u32,
                            descriptor_type: *descriptor_type,
                            descriptor_count: *descriptor_count,
                            stage_flags: ShaderStageFlagBits::All as u32,
                            p_immutable_samplers: ptr::null(),
                        },
                    )
                    .collect::<Vec<_>>();
                dbg!(bindings.len());

                let bind_flag = [
                    DescriptorBindingFlagBits::PartiallyBound,
                    DescriptorBindingFlagBits::UpdateAfterBind,
                ]
                .into_iter()
                .fold(0, |accum, x| accum | x as u32);
                let bind_flags = vec![bind_flag; bindings.len()];

                let binding_flags = DescriptorSetLayoutBindingFlagsCreateInfo {
                    s_type: StructureType::DescriptorSetLayoutBindingFlagsCreateInfo,
                    p_next: ptr::null(),
                    binding_count: bindings.len() as u32,
                    flags: bind_flags.as_ptr(),
                };

                let flags = [DescriptorSetLayoutCreateFlagBits::UpdateAfterBindPool]
                    .into_iter()
                    .fold(0, |accum, x| accum | x as u32);

                let descriptor_set_layout_info = DescriptorSetLayoutCreateInfo {
                    s_type: StructureType::DescriptorSetLayoutCreateInfo,
                    p_next: &binding_flags as *const _ as *const _,
                    flags,
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                };

                let mut descriptor_set_layout = Self::default();
                VkResult::handle(unsafe {
                    vkCreateDescriptorSetLayout(
                        device,
                        &descriptor_set_layout_info,
                        ptr::null(),
                        &mut descriptor_set_layout,
                    )
                })?;
                Ok(descriptor_set_layout)
            }
        }

        pub struct ShaderSource(Vec<u8>);

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
            fn compile(
                device: Device,
                set_layouts: &[DescriptorSetLayout],
                spec: Vec<ShaderSpec>,
            ) -> Result<Vec<Self>, Error> {
                let source = spec
                    .iter()
                    .map(|spec| spec.source.0.clone())
                    .collect::<Vec<_>>();

                let name = spec
                    .iter()
                    .map(|spec| spec.entry.as_ref())
                    .map(CString::from_str)
                    .map(|x| x.unwrap())
                    .collect::<Vec<_>>();

                let info = (0..spec.len())
                    .map(|i| ShaderCreateInfo {
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
                    })
                    .collect::<Vec<_>>();

                let mut shaders = vec![Self::default(); info.len()];

                let vkCreateShadersEXT = unsafe {
                    mem::transmute::<_, vkCreateShadersEXT>(load_device_fn(
                        device,
                        VK_CREATE_SHADERS_EXT,
                    ))
                };

                VkResult::handle(unsafe {
                    dbg!((vkCreateShadersEXT)(
                        device,
                        info.len() as u32,
                        info.as_ptr(),
                        ptr::null(),
                        shaders.as_mut_ptr(),
                    ))
                })?;

                Ok(shaders)
            }
        }

        unsafe fn load_device_fn(device: Device, name: &str) -> fn() {
            let compile_name = CString::from_str(name).unwrap();
            vkGetDeviceProcAddr(device, compile_name.as_ref().as_ptr())
        }

        #[derive(Debug)]
        pub enum Error {
            Unknown,
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
                let (major, minor, patch) = self;
                ((major as u32) << 22) | ((minor as u32) << 12) | (patch as u32)
            }
        }

        fn c_str_array(
            array: &[impl AsRef<str>],
        ) -> (Vec<CString>, Vec<*const c_char>, *const *const c_char) {
            let c_strings = array
                .iter()
                .map(AsRef::as_ref)
                .map(CString::from_str)
                .map(|x| x.unwrap())
                .collect::<Vec<_>>();
            let c_ptrs = c_strings
                .iter()
                .map(CString::as_ref)
                .map(CStr::as_ptr)
                .collect::<Vec<_>>();
            let c_ptr_ptr = c_ptrs.as_ptr();
            (c_strings, c_ptrs, c_ptr_ptr)
        }

        pub fn convert_c_str<const N: usize>(mut array: [u8; N]) -> String {
            let mut v = array.to_vec();
            v.truncate(
                array
                    .iter()
                    .enumerate()
                    .find_map(|(i, &x)| (x == 0).then_some(i))
                    .unwrap(),
            );
            let string_c = unsafe { CString::from_vec_unchecked(v) };
            string_c.to_str().unwrap().to_string()
        }

        #[repr(u32)]
        pub enum StructureType {
            ApplicationInfo = 0,
            InstanceCreateInfo = 1,
            DeviceQueueInfo = 2,
            DeviceInfo = 3,
            SubmitInfo = 4,
            MemoryAllocateInfo = 6,
            FenceCreateInfo = 8,
            SemaphoreCreateInfo = 9,
            BufferCreateInfo = 12,
            ImageViewCreateInfo = 15,
            PipelineLayoutCreateInfo = 30,
            DescriptorSetLayoutCreateInfo = 32,
            DescriptorPoolCreateInfo = 33,
            DescriptorSetAllocateInfo = 34,
            CommandPoolCreateInfo = 39,
            CommandBufferAllocateInfo = 40,
            CommandBufferBeginInfo = 42,
            BufferMemoryBarrier = 44,
            ImageMemoryBarrier = 45,
            PhysicalDeviceFeatures2 = 1000059000,
            BufferDeviceAddressInfo = 1000244001,
            PhysicalDeviceDescriptorIndexingFeatures = 1000161001,
            PhysicalDeviceDynamicRenderingFeatures = 1000044003,
            PhysicalDeviceBufferDeviceAddressFeatures = 1000257000,
            PresentInfo = 1000001001,
            MemoryAllocateFlagsInfo = 1000060000,
            SwapchainCreateInfo = 1000001000,
            XlibSurfaceCreateInfo = 1000004000,
            RenderingInfo = 1000044000,
            RenderingAttachmentInfo = 1000044001,
            ShaderCreateInfo = 1000482002,
            DescriptorSetLayoutBindingFlagsCreateInfo = 1000161000,
            PhysicalDeviceShaderObjectFeatures = 1000482000,
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
        pub enum SampleCountFlagBits {
            _1 = 0x00000001,
            _2 = 0x00000002,
            _4 = 0x00000004,
            _8 = 0x00000008,
            _16 = 0x00000010,
            _32 = 0x00000020,
            _64 = 0x00000040,
        }

        #[repr(C)]
        pub struct PhysicalDeviceShaderObjectFeatures {
            s_type: StructureType,
            p_next: *const c_void,
            shader_object: Bool32,
        }

        #[repr(C)]
        pub struct PhysicalDeviceDynamicRenderingFeatures {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub dynamic_rendering: Bool32,
        }

        #[repr(C)]
        pub struct PhysicalDeviceBufferDeviceAddressFeatures {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub buffer_device_address: Bool32,
            pub buffer_device_address_capture_replay: Bool32,
            pub buffer_device_address_multi_device: Bool32,
        }

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
                    e => Err(Error::Unknown),
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
            Bgra8Unorm = 44,
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

        #[repr(C)]
        #[derive(Debug, Copy, Clone)]
        pub struct Image(*const c_void);
        impl Default for Image {
            fn default() -> Self {
                Image(ptr::null())
            }
        }
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
            fn map(self, device: Device, offset: u64, size: u64) -> *mut c_void {
                let mut ptr = ptr::null_mut();
                unsafe { vkMapMemory(device, self, offset, size, 0, &mut ptr) };
                ptr
            }
            fn unmap(self, device: Device) {
                unsafe { vkUnmapMemory(device, self) };
            }
            fn host(
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
            fn device_local(
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

            fn bind_buffer(self, device: Device, buffer: Buffer) -> Self {
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

        #[repr(C)]
        pub struct PhysicalDeviceMemoryProperties {
            memory_type_count: u32,
            memory_types: [MemoryType; VK_MAX_MEMORY_TYPES],
            memory_heap_count: u32,
            memory_heaps: [MemoryHeap; VK_MAX_MEMORY_HEAPS],
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
                        && (mem_properties.memory_types[*i].property_flags & properties
                            == properties)
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

        impl Buffer {
            fn create(device: Device, size: DeviceSize, usage: BufferUsageFlags) -> Self {
                let info = BufferCreateInfo {
                    s_type: StructureType::BufferCreateInfo,
                    p_next: ptr::null(),
                    flags: 0,
                    size,
                    usage: usage | BufferUsageFlagBits::ShaderDeviceAddress as u32,
                    sharing_mode: SharingMode::Exclusive,
                    queue_family_index_count: 0,
                    p_queue_family_indices: ptr::null(),
                };
                let mut buffer = Self::default();
                unsafe { vkCreateBuffer(device, &info, ptr::null(), &mut buffer) };
                buffer
            }

            fn requirements(self, device: Device) -> MemoryRequirements {
                let mut req = unsafe { mem::zeroed::<MemoryRequirements>() };
                unsafe { vkGetBufferMemoryRequirements(device, self, &mut req) };
                req
            }

            fn device_address(self, device: Device) -> DeviceAddress {
                let info = BufferDeviceAddressInfo {
                    s_type: StructureType::BufferDeviceAddressInfo,
                    p_next: ptr::null(),
                    buffer: self,
                };
                unsafe { vkGetBufferDeviceAddress(device, &info) }
            }
        }

        #[repr(C)]
        pub struct BufferDeviceAddressInfo {
            s_type: StructureType,
            p_next: *const c_void,
            buffer: Buffer,
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
            fn create(
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

        #[repr(C)]
        pub struct RenderingInfo {
            s_type: StructureType,
            p_next: *const c_void,
            flags: RenderingFlags,
            render_area: Rect2D,
            layer_count: u32,
            view_mask: u32,
            color_attachment_count: u32,
            p_color_attachments: *const RenderingAttachmentInfo,
            p_depth_attachment: *const RenderingAttachmentInfo,
            p_stencil_attachment: *const RenderingAttachmentInfo,
        }

        #[repr(C)]
        pub struct Rect2D {
            offset: Offset2D,
            extent: Extent2D,
        }

        #[repr(C)]
        pub struct Offset2D {
            x: i32,
            y: i32,
        }

        #[repr(C)]
        pub struct RenderingAttachmentInfo {
            s_type: StructureType,
            p_next: *const c_void,
            image_view: ImageView,
            image_layout: ImageLayout,
            resolve_mode: ResolveMode,
            resolve_image_view: ImageView,
            resolve_image_layout: ImageLayout,
            load_op: AttachmentLoadOp,
            store_op: AttachmentStoreOp,
            clear_value: ClearValue,
        }

        pub type RenderingFlags = u32;

        #[repr(u32)]
        pub enum RenderingFlagBits {
            ContentsSecondaryCommandBuffers = 0x00000001,
            Suspending = 0x00000002,
            Resuming = 0x00000004,
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
        pub enum ResolveMode {
            None = 0,
            SampleZero = 1,
            Average = 2,
            Min = 3,
            Max = 4,
        }

        #[repr(C)]
        pub enum AttachmentLoadOp {
            Load = 0,
            Clear = 1,
            DontCare = 2,
        }

        #[repr(C)]
        pub enum AttachmentStoreOp {
            Store = 0,
            DontCare = 1,
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub union ClearValue {
            color: ClearColorValue,
            depth_stencil: ClearDepthStencilValue,
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub union ClearColorValue {
            float32: [f32; 4],
            int32: [i32; 4],
            uint32: [u32; 4],
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct ClearDepthStencilValue {
            depth: f32,
            stencil: u32,
        }

        pub struct RenderAttachment {
            image_view: ImageView,
            image_layout: ImageLayout,
            load_op: AttachmentLoadOp,
            store_op: AttachmentStoreOp,
            clear_value: ClearValue,
        }

        pub struct Render {
            width: u32,
            height: u32,
            color: Vec<RenderAttachment>,
            depth: Option<RenderAttachment>,
            stencil: Option<RenderAttachment>,
        }

        #[repr(C)]
        pub struct CommandBufferBeginInfo {
            s_type: StructureType,
            p_next: *const c_void,
            flags: CommandBufferUsageFlags,
            p_inheritance_info: Option<*const c_void>,
        }

        pub type CommandBufferUsageFlags = u32;

        #[repr(u32)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum CommandBufferUsageFlagBits {
            OneTimeSubmit = 0x00000001,
            RenderPassContinue = 0x00000002,
            SimultaneousUse = 0x00000004,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct Viewport {
            pub x: f32,
            pub y: f32,
            pub width: f32,
            pub height: f32,
            pub min_depth: f32,
            pub max_depth: f32,
        }

        #[repr(C)]
        pub enum PrimitiveTopology {
            PointList = 0,
            LineList = 1,
            LineStrip = 2,
            TriangleList = 3,
            TriangleStrip = 4,
            TriangleFan = 5,
            // ... other variants
        }

        #[repr(i32)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum PolygonMode {
            Fill = 0,
            Line = 1,
            Point = 2,
            FillRectangleNV = 1000153000,
        }

        pub type ColorComponentFlags = u32;

        #[repr(C)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum ColorComponentFlagBits {
            R = 0x00000001,
            G = 0x00000002,
            B = 0x00000004,
            A = 0x00000008,
        }

        #[repr(C)]
        #[derive(Copy, Clone, Debug)]
        pub struct ColorBlendEquationEXT {
            pub src_color_blend_factor: BlendFactor,
            pub dst_color_blend_factor: BlendFactor,
            pub color_blend_op: BlendOp,
            pub src_alpha_blend_factor: BlendFactor,
            pub dst_alpha_blend_factor: BlendFactor,
            pub alpha_blend_op: BlendOp,
        }

        pub struct SampleMask(*const c_void);
        impl Default for SampleMask {
            fn default() -> Self {
                Self(ptr::null())
            }
        }

        #[repr(i32)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum BlendFactor {
            Zero = 0,
            One = 1,
            SrcColor = 2,
            OneMinusSrcColor = 3,
            DstColor = 4,
            OneMinusDstColor = 5,
            SrcAlpha = 6,
            OneMinusSrcAlpha = 7,
            DstAlpha = 8,
            OneMinusDstAlpha = 9,
            ConstantColor = 10,
            OneMinusConstantColor = 11,
            ConstantAlpha = 12,
            OneMinusConstantAlpha = 13,
            SrcAlphaSaturate = 14,
            Src1Color = 15,
            OneMinusSrc1Color = 16,
            Src1Alpha = 17,
            OneMinusSrc1Alpha = 18,
        }

        #[repr(i32)]
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub enum BlendOp {
            Add = 0,
            Subtract = 1,
            ReverseSubtract = 2,
            Min = 3,
            Max = 4,
        }

        #[repr(C)]
        pub enum FrontFace {
            CounterClockwise = 0,
            Clockwise = 1,
        }

        #[repr(C)]
        pub enum CullModeFlagBits {
            None = 0,
            Front = 0x00000001,
            Back = 0x00000002,
            FrontAndBack = 0x00000003,
        }

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
            fn create_signaled(device: Device) -> Result<Self, Error> {
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
            fn wait(self, device: Device) -> bool {
                VkResult::handle(unsafe { vkWaitForFences(device, 1, &self, true as _, 0) })
                    .is_err()
            }
            fn reset(self, device: Device) {
                unsafe { vkResetFences(device, 1, &self) };
            }
        }

        #[repr(C)]
        pub enum PipelineStageFlagBits {
            TopOfPipe = 0x00000001,
            DrawIndirect = 0x00000002,
            VertexInput = 0x00000004,
            VertexShader = 0x00000008,
            TessellationControlShader = 0x00000010,
            TessellationEvaluationShader = 0x00000020,
            GeometryShader = 0x00000040,
            FragmentShader = 0x00000080,
            EarlyFragmentTests = 0x00000100,
            LateFragmentTests = 0x00000200,
            ColorAttachmentOutput = 0x00000400,
            ComputeShader = 0x00000800,
            Transfer = 0x00001000,
            BottomOfPipe = 0x00002000,
            Host = 0x00004000,
            AllGraphics = 0x00008000,
            AllCommands = 0x00010000,
        }

        pub type PipelineStageFlags = u32;

        #[repr(C)]
        pub struct SubmitInfo {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub wait_semaphore_count: u32,
            pub p_wait_semaphores: *const Semaphore,
            pub p_wait_dst_stage_mask: *const PipelineStageFlags,
            pub command_buffer_count: u32,
            pub p_command_buffers: *const CommandBuffer,
            pub signal_semaphore_count: u32,
            pub p_signal_semaphores: *const Semaphore,
        }

        #[repr(C)]
        pub struct PresentInfo {
            pub s_type: StructureType,
            pub p_next: *const c_void,
            pub wait_semaphore_count: u32,
            pub p_wait_semaphores: *const Semaphore,
            pub swapchain_count: u32,
            pub p_swapchains: *const Swapchain,
            pub p_image_indices: *const u32,
            pub p_results: *mut VkResult,
        }

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Queue(*const c_void);

        impl Default for Queue {
            fn default() -> Self {
                Self(ptr::null())
            }
        }

        impl Queue {
            pub fn submit(
                self,
                wait: Semaphore,
                signal: Semaphore,
                fence: Fence,
                cmd: CommandBuffer,
            ) {
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

        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct Semaphore(*const c_void);
        impl Default for Semaphore {
            fn default() -> Self {
                Self(ptr::null())
            }
        }
        impl Semaphore {
            fn create(device: Device) -> Result<Self, Error> {
                let info = SemaphoreCreateInfo {
                    s_type: StructureType::SemaphoreCreateInfo,
                    p_next: ptr::null(),
                    flags: 0,
                };
                let mut semaphore = Self::default();
                unsafe { vkCreateSemaphore(device, &info, ptr::null(), &mut semaphore) };
                Ok(semaphore)
            }
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

        #[repr(C)]
        pub enum SurfaceTransformFlagBitsKHR {
            IdentityKHR = 0x00000001,
            Rotate90KHR = 0x00000002,
            Rotate180KHR = 0x00000004,
            Rotate270KHR = 0x00000008,
            HorizontalMirrorKHR = 0x00000010,
            HorizontalMirrorRotate90KHR = 0x00000020,
            HorizontalMirrorRotate180KHR = 0x00000040,
            HorizontalMirrorRotate270KHR = 0x00000080,
            InheritKHR = 0x00000100,
        }

        pub type CompositeAlphaFlagsKHR = u32;
        pub type SurfaceTransformFlagsKHR = u32;

        #[repr(C)]
        pub enum CompositeAlphaFlagBitsKHR {
            OpaqueKHR = 0x00000001,
            PreMultipliedKHR = 0x00000002,
            PostMultipliedKHR = 0x00000004,
            InheritKHR = 0x00000008,
        }

        pub type CullModeFlags = u32;

        #[repr(C)]
        pub struct BufferCopy {
            src_offset: DeviceSize,
            dst_offset: DeviceSize,
            size: DeviceSize,
        }

        type vkCmdBindShadersEXT = fn(
            cmd: CommandBuffer,
            stage_count: u32,
            stages: *const ShaderStageFlags,
            shaders: *const Shader,
        );
        const VK_CMD_BIND_SHADERS_EXT: &str = "vkCmdBindShadersEXT";

        type vkCreateShadersEXT = fn(
            device: Device,
            count: u32,
            infos: *const ShaderCreateInfo,
            alloc: *const c_void,
            shaders: *mut Shader,
        ) -> VkResult;
        const VK_CREATE_SHADERS_EXT: &str = "vkCreateShadersEXT";

        type vkCmdSetPolygonModeEXT = fn(command_buffer: CommandBuffer, polygon_mode: PolygonMode);
        const VK_CMD_SET_POLYGON_MODE_EXT: &str = "vkCmdSetPolygonModeEXT";

        type vkCmdSetRasterizationSamplesEXT =
            fn(command_buffer: CommandBuffer, rasterization_samples: SampleCountFlags);
        const VK_CMD_SET_RASTERIZATION_SAMPLES_EXT: &str = "vkCmdSetRasterizationSamplesEXT";

        type vkCmdSetSampleMaskEXT = fn(
            command_buffer: CommandBuffer,
            samples: SampleCountFlags,
            p_sample_mask: *const SampleMask,
        );
        const VK_CMD_SET_SAMPLE_MASK_EXT: &str = "vkCmdSetSampleMaskEXT";

        type vkCmdSetAlphaToCoverageEnableEXT =
            fn(command_buffer: CommandBuffer, alpha_to_coverage_enable: Bool32);
        const VK_CMD_SET_ALPHA_TO_COVERAGE_ENABLE_EXT: &str = "vkCmdSetAlphaToCoverageEnableEXT";

        type vkCmdSetColorBlendEnableEXT = fn(
            command_buffer: CommandBuffer,
            first_attachment: u32,
            attachment_count: u32,
            p_color_blend_enables: *const Bool32,
        );
        const VK_CMD_SET_COLOR_BLEND_ENABLE_EXT: &str = "vkCmdSetColorBlendEnableEXT";

        type vkCmdSetColorBlendEquationEXT = fn(
            command_buffer: CommandBuffer,
            first_attachment: u32,
            attachment_count: u32,
            p_color_blend_equations: *const ColorBlendEquationEXT,
        );
        const VK_CMD_SET_COLOR_BLEND_EQUATION_EXT: &str = "vkCmdSetColorBlendEquationEXT";

        type vkCmdSetColorWriteMaskEXT = fn(
            command_buffer: CommandBuffer,
            first_attachment: u32,
            attachment_count: u32,
            p_color_write_masks: *const ColorComponentFlags,
        );
        const VK_CMD_SET_COLOR_WRITE_MASK_EXT: &str = "vkCmdSetColorWriteMaskEXT";

        extern "C" {
            pub fn vkCreateInstance(
                info: *const InstanceCreateInfo,
                alloc: *const c_void,
                out: *mut Instance,
            ) -> VkResult;
            pub fn vkEnumeratePhysicalDevices(
                instance: Instance,
                count: *mut u32,
                physical_devices: *mut PhysicalDevice,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceProperties(
                physical_device: PhysicalDevice,
                property: *mut PhysicalDeviceProperties,
            );
            pub fn vkGetDeviceQueue(
                device: Device,
                queue_family_index: QueueFamilyIndex,
                queue_index: u32,
                queue: *mut Queue,
            ) -> VkResult;
            pub fn vkCreateDevice(
                physical_device: PhysicalDevice,
                info: *const DeviceCreateInfo,
                alloc: *const c_void,
                out: *mut Device,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device: PhysicalDevice,
                qfi: u32,
                surface: Surface,
                support: *mut Bool32,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device: PhysicalDevice,
                count: *mut u32,
                physical_devices: *mut QueueFamilyProperties,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                physical_device: PhysicalDevice,
                surface: Surface,
                capabilities: *mut SurfaceCapabilitiesKHR,
            ) -> VkResult;
            pub fn vkCreateXlibSurfaceKHR(
                instance: Instance,
                info: *const XlibSurfaceCreateInfo,
                alloc: *const c_void,
                surface: *mut Surface,
            ) -> VkResult;
            pub fn vkCreatePipelineLayout(
                device: Device,
                info: *const PipelineLayoutCreateInfo,
                alloc: *const c_void,
                pipeline_layout: *mut PipelineLayout,
            ) -> VkResult;
            pub fn vkCreateSemaphore(
                device: Device,
                info: *const SemaphoreCreateInfo,
                alloc: *const c_void,
                image_view: *mut Semaphore,
            ) -> VkResult;
            pub fn vkCreateBuffer(
                device: Device,
                info: *const BufferCreateInfo,
                alloc: *const c_void,
                buffer: *mut Buffer,
            ) -> VkResult;
            pub fn vkGetBufferDeviceAddress(
                device: Device,
                info: *const BufferDeviceAddressInfo,
            ) -> DeviceAddress;
            pub fn vkCreateFence(
                device: Device,
                info: *const FenceCreateInfo,
                alloc: *const c_void,
                image_view: *mut Fence,
            ) -> VkResult;
            pub fn vkMapMemory(
                device: Device,
                memory: Memory,
                offset: DeviceSize,
                size: DeviceSize,
                flags: u32,
                ptr: *mut *mut c_void,
            ) -> VkResult;
            pub fn vkUnmapMemory(device: Device, memory: Memory);
            pub fn vkAllocateMemory(
                device: Device,
                info: *const MemoryAllocateInfo,
                alloc: *const c_void,
                mem: *mut Memory,
            ) -> VkResult;
            pub fn vkBindBufferMemory(
                device: Device,
                buffer: Buffer,
                memory: Memory,
                size: DeviceSize,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceMemoryProperties(
                physical_device: PhysicalDevice,
                properties: *mut PhysicalDeviceMemoryProperties,
            );
            pub fn vkGetBufferMemoryRequirements(
                device: Device,
                buffer: Buffer,
                req: *mut MemoryRequirements,
            );
            pub fn vkResetFences(device: Device, count: u32, fences: *const Fence) -> VkResult;
            pub fn vkWaitForFences(
                device: Device,
                count: u32,
                fences: *const Fence,
                wait_all: Bool32,
                timeout: u64,
            ) -> VkResult;
            pub fn vkCreateImageView(
                device: Device,
                info: *const ImageViewCreateInfo,
                alloc: *const c_void,
                image_view: *mut ImageView,
            ) -> VkResult;
            pub fn vkCreateSwapchainKHR(
                device: Device,
                info: *const SwapchainCreateInfo,
                alloc: *const c_void,
                swapchain: *mut Swapchain,
            ) -> VkResult;
            pub fn vkGetSwapchainImagesKHR(
                device: Device,
                swapchain: Swapchain,
                image_count: *mut u32,
                images: *mut Image,
            );
            pub fn vkCreateDescriptorPool(
                device: Device,
                info: *const DescriptorPoolCreateInfo,
                alloc: *const c_void,
                descriptor_pool: *mut DescriptorPool,
            ) -> VkResult;
            pub fn vkCreateDescriptorSetLayout(
                device: Device,
                info: *const DescriptorSetLayoutCreateInfo,
                alloc: *const c_void,
                descriptor_pool: *mut DescriptorSetLayout,
            ) -> VkResult;
            pub fn vkGetDeviceProcAddr(device: Device, name: *const c_char) -> fn();
            pub fn vkAllocateDescriptorSets(
                device: Device,
                info: *const DescriptorSetAllocateInfo,
                descriptor_sets: *mut DescriptorSet,
            ) -> VkResult;
            pub fn vkCreateCommandPool(
                device: Device,
                info: *const CommandPoolCreateInfo,
                alloc: *const c_void,
                swapchain: *mut CommandPool,
            ) -> VkResult;
            pub fn vkAllocateCommandBuffers(
                device: Device,
                info: *const CommandBufferAllocateInfo,
                descriptor_sets: *mut CommandBuffer,
            ) -> VkResult;
            pub fn vkBeginCommandBuffer(cmd: CommandBuffer, info: *const CommandBufferBeginInfo);
            pub fn vkEndCommandBuffer(cmd: CommandBuffer);
            pub fn vkCmdBeginRendering(cmd: CommandBuffer, info: *const RenderingInfo);
            pub fn vkCmdEndRendering(cmd: CommandBuffer);
            pub fn vkCmdCopyBuffer(
                cmd: CommandBuffer,
                src: Buffer,
                dst: Buffer,
                count: u32,
                regions: *const BufferCopy,
            );
            pub fn vkCmdPushConstants(
                cmd: CommandBuffer,
                layout: PipelineLayout,
                stage_flags: ShaderStageFlags,
                offset: u32,
                size: u32,
                data: *const c_void,
            );
            pub fn vkCmdDraw(
                cmd: CommandBuffer,
                vertex_count: u32,
                instance_count: u32,
                first_vertex: u32,
                first_instance: u32,
            );
            pub fn vkCmdSetViewportWithCount(
                command_buffer: CommandBuffer,
                viewport_count: u32,
                p_viewports: *const Viewport,
            );
            pub fn vkAcquireNextImageKHR(
                device: Device,
                swapchain: Swapchain,
                timeout: u64,
                semaphore: Semaphore,
                fence: Fence,
                image_index: *mut u32,
            ) -> VkResult;
            fn vkCmdPipelineBarrier(
                command_buffer: CommandBuffer,
                src_stage_mask: PipelineStageFlags,
                dst_stage_mask: PipelineStageFlags,
                dependency_flags: DependencyFlags,
                memory_barrier_count: u32,
                p_memory_barriers: *const MemoryBarrier,
                buffer_memory_barrier_count: u32,
                p_buffer_memory_barriers: *const BufferMemoryBarrier,
                image_memory_barrier_count: u32,
                p_image_memory_barriers: *const ImageMemoryBarrier,
            );
            pub fn vkCmdSetScissorWithCount(
                command_buffer: CommandBuffer,
                scissor_count: u32,
                p_scissors: *const Rect2D,
            );
            pub fn vkCmdSetRasterizerDiscardEnable(
                command_buffer: CommandBuffer,
                rasterizer_discard_enable: Bool32,
            );
            pub fn vkCmdSetPrimitiveTopology(
                command_buffer: CommandBuffer,
                primitive_topology: PrimitiveTopology,
            );
            pub fn vkCmdSetPrimitiveRestartEnable(
                command_buffer: CommandBuffer,
                primitive_restart_enable: Bool32,
            );
            pub fn vkCmdSetCullMode(command_buffer: CommandBuffer, cull_mode: CullModeFlags);

            pub fn vkCmdSetFrontFace(command_buffer: CommandBuffer, front_face: FrontFace);

            pub fn vkCmdSetDepthTestEnable(
                command_buffer: CommandBuffer,
                depth_test_enable: Bool32,
            );

            pub fn vkCmdSetDepthWriteEnable(
                command_buffer: CommandBuffer,
                depth_write_enable: Bool32,
            );

            pub fn vkCmdSetDepthBiasEnable(
                command_buffer: CommandBuffer,
                depth_bias_enable: Bool32,
            );

            pub fn vkCmdSetStencilTestEnable(
                command_buffer: CommandBuffer,
                stencil_test_enable: Bool32,
            );

            pub fn vkQueueSubmit(
                queue: Queue,
                submit_count: u32,
                submits: *const SubmitInfo,
                fence: Fence,
            ) -> VkResult;
            pub fn vkQueuePresentKHR(queue: Queue, present: *const PresentInfo) -> VkResult;
        }

        mod cmd {
            use core::{mem, ptr};
            use mem::transmute;

            use crate::gfx::vk::{
                AccessFlags, BlendFactor, BlendOp, Buffer,
                BufferCopy, BufferMemoryBarrier, ColorBlendEquationEXT, ColorComponentFlagBits,
                CommandBuffer, CommandBufferBeginInfo, CullModeFlagBits,
                Device, DeviceSize, Extent2D,
                FrontFace, Image, ImageLayout,
                ImageMemoryBarrier, ImageSubresourceRange, ImageView,
                load_device_fn, Offset2D,
                PipelineLayout, PipelineStageFlags, PolygonMode,
                PrimitiveTopology, PushConstant, Rect2D,
                Render, RenderAttachment, RenderingAttachmentInfo, RenderingInfo, ResolveMode, SampleCountFlagBits,
                SampleMask, Shader, ShaderStageFlagBits,
                ShaderStageFlags, StructureType, Viewport, VK_CMD_BIND_SHADERS_EXT, VK_CMD_SET_ALPHA_TO_COVERAGE_ENABLE_EXT, VK_CMD_SET_COLOR_BLEND_ENABLE_EXT,
                VK_CMD_SET_COLOR_BLEND_EQUATION_EXT, VK_CMD_SET_COLOR_WRITE_MASK_EXT, VK_CMD_SET_POLYGON_MODE_EXT, VK_CMD_SET_RASTERIZATION_SAMPLES_EXT, VK_CMD_SET_SAMPLE_MASK_EXT, vkBeginCommandBuffer,
                vkCmdBeginRendering, vkCmdBindShadersEXT, vkCmdCopyBuffer, vkCmdDraw, vkCmdEndRendering,
                vkCmdPipelineBarrier, vkCmdPushConstants, vkCmdSetAlphaToCoverageEnableEXT, vkCmdSetColorBlendEnableEXT, vkCmdSetColorBlendEquationEXT,
                vkCmdSetColorWriteMaskEXT, vkCmdSetCullMode, vkCmdSetDepthBiasEnable, vkCmdSetDepthTestEnable, vkCmdSetDepthWriteEnable,
                vkCmdSetFrontFace, vkCmdSetPolygonModeEXT, vkCmdSetPrimitiveRestartEnable, vkCmdSetPrimitiveTopology,
                vkCmdSetRasterizationSamplesEXT, vkCmdSetRasterizerDiscardEnable,
                vkCmdSetSampleMaskEXT, vkCmdSetScissorWithCount,
                vkCmdSetStencilTestEnable, vkCmdSetViewportWithCount,
                vkEndCommandBuffer,
            };
            use crate::prelude::*;

            pub(crate) fn begin(cmd: CommandBuffer) {
                let info = CommandBufferBeginInfo {
                    s_type: StructureType::CommandBufferBeginInfo,
                    p_next: ptr::null(),
                    flags: 0,
                    p_inheritance_info: None,
                };
                unsafe { vkBeginCommandBuffer(cmd, &info) };
            }

            pub(crate) fn end(cmd: CommandBuffer) {
                unsafe { vkEndCommandBuffer(cmd) };
            }

            pub(crate) fn begin_render(cmd: CommandBuffer, render: Render) {
                let Render { width, height, .. } = render;
                let convert = |a: RenderAttachment| RenderingAttachmentInfo {
                    s_type: StructureType::RenderingAttachmentInfo,
                    p_next: ptr::null(),
                    image_view: a.image_view,
                    image_layout: a.image_layout,
                    resolve_mode: ResolveMode::None,
                    resolve_image_view: ImageView::default(),
                    resolve_image_layout: ImageLayout::Undefined,
                    load_op: a.load_op,
                    store_op: a.store_op,
                    clear_value: a.clear_value,
                };
                let color = render.color.into_iter().map(convert).collect::<Vec<_>>();
                let depth = render.depth.map(convert);
                let stencil = render.stencil.map(convert);
                let info = RenderingInfo {
                    s_type: StructureType::RenderingInfo,
                    p_next: ptr::null(),
                    flags: 0,
                    render_area: Rect2D {
                        offset: Offset2D { x: 0, y: 0 },
                        extent: Extent2D { width, height },
                    },
                    layer_count: 1,
                    view_mask: 0,
                    color_attachment_count: color.len() as u32,
                    p_color_attachments: color.as_ptr(),
                    p_depth_attachment: depth
                        .as_ref()
                        .map(|x| x as *const _)
                        .unwrap_or(ptr::null()),
                    p_stencil_attachment: stencil
                        .as_ref()
                        .map(|x| x as *const _)
                        .unwrap_or(ptr::null()),
                };
                unsafe { vkCmdBeginRendering(cmd, &info) };
            }

            pub fn copy_buffer(
                cmd: CommandBuffer,
                src: Buffer,
                dst: Buffer,
                src_offset: u64,
                dst_offset: u64,
                size: u64,
            ) {
                let copy = BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                };
                unsafe {
                    vkCmdCopyBuffer(
                        cmd,
                        src,
                        dst,
                        1,
                        &copy,
                    )
                }
            }

            pub(crate) fn end_render(cmd: CommandBuffer) {
                unsafe { vkCmdEndRendering(cmd) };
            }

            pub fn push_constant<T: Copy>(
                cmd: CommandBuffer,
                layout: PipelineLayout,
                push: PushConstant<T>,
            ) {
                unsafe {
                    vkCmdPushConstants(
                        cmd,
                        layout,
                        ShaderStageFlagBits::All as u32,
                        push.offset,
                        mem::size_of::<T>() as u32,
                        &push.data as *const _ as *const _,
                    )
                };
            }

            #[derive(Clone, Copy, Debug)]
            pub struct BufferBarrier {
                pub src_access: AccessFlags,
                pub dst_access: AccessFlags,
                pub buffer: Buffer,
                pub offset: DeviceSize,
                pub size: DeviceSize,
            }

            pub fn pipeline_barrier_buffer(
                cmd: CommandBuffer,
                src_stage: PipelineStageFlags,
                dst_stage: PipelineStageFlags,
                barriers: &[BufferBarrier],
            ) {
                let barriers = barriers
                    .iter()
                    .copied()
                    .map(
                        |BufferBarrier {
                             src_access: src_access_mask,
                             dst_access: dst_access_mask,
                             buffer,
                             offset,
                             size,
                         }| BufferMemoryBarrier {
                            s_type: StructureType::BufferMemoryBarrier,
                            p_next: ptr::null(),
                            src_access_mask,
                            dst_access_mask,
                            src_queue_family_index: 0,
                            dst_queue_family_index: 0,
                            buffer,
                            offset,
                            size,
                        },
                    )
                    .collect::<Vec<_>>();
                unsafe {
                    vkCmdPipelineBarrier(
                        cmd,
                        src_stage,
                        dst_stage,
                        0,
                        0,
                        ptr::null(),
                        barriers.len() as u32,
                        barriers.as_ptr(),
                        0,
                        ptr::null(),
                    )
                }
            }

            pub fn pipeline_barrier_image(
                cmd: CommandBuffer,
                src_access_mask: AccessFlags,
                src_stage_mask: PipelineStageFlags,
                dst_access_mask: AccessFlags,
                dst_stage_mask: PipelineStageFlags,
                transitions: Vec<(Image, ImageLayout, ImageLayout)>,
            ) {
                let transitions = transitions
                    .into_iter()
                    .map(|(image, old_layout, new_layout)| ImageMemoryBarrier {
                        s_type: StructureType::ImageMemoryBarrier,
                        p_next: ptr::null(),
                        src_access_mask,
                        dst_access_mask,
                        old_layout,
                        new_layout,
                        src_queue_family_index: 0,
                        dst_queue_family_index: 0,
                        image,
                        subresource_range: ImageSubresourceRange {
                            aspect_mask: 1,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                    })
                    .collect::<Vec<_>>();
                unsafe {
                    vkCmdPipelineBarrier(
                        cmd,
                        src_stage_mask,
                        dst_stage_mask,
                        0,
                        0,
                        ptr::null(),
                        0,
                        ptr::null(),
                        transitions.len() as u32,
                        transitions.as_ptr(),
                    )
                }
            }

            pub(crate) fn bind_shaders(
                device: Device,
                cmd: CommandBuffer,
                stages: Vec<ShaderStageFlags>,
                shaders: Vec<Shader>,
            ) {
                let vkCmdBindShadersEXT = unsafe {
                    transmute::<_, vkCmdBindShadersEXT>(load_device_fn(
                        device,
                        VK_CMD_BIND_SHADERS_EXT,
                    ))
                };
                unsafe {
                    (vkCmdBindShadersEXT)(
                        cmd,
                        shaders.len() as u32,
                        stages.as_ptr(),
                        shaders.as_ptr(),
                    )
                };
            }

            pub(crate) fn draw_settings(
                device: Device,
                cmd: CommandBuffer,
                width: u32,
                height: u32,
            ) {
                unsafe {
                    vkCmdSetViewportWithCount(
                        cmd,
                        1,
                        &Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: width as f32,
                            height: height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        },
                    );
                    vkCmdSetScissorWithCount(
                        cmd,
                        1,
                        &Rect2D {
                            offset: Offset2D { x: 0, y: 0 },
                            extent: Extent2D { width, height },
                        },
                    );
                    vkCmdSetPrimitiveTopology(cmd, PrimitiveTopology::TriangleList);
                    vkCmdSetRasterizerDiscardEnable(cmd, false as _);
                    vkCmdSetPrimitiveRestartEnable(cmd, false as _);
                    vkCmdSetCullMode(cmd, CullModeFlagBits::None as u32);

                    vkCmdSetFrontFace(cmd, FrontFace::CounterClockwise);
                    vkCmdSetDepthTestEnable(cmd, false as _);

                    vkCmdSetDepthWriteEnable(cmd, false as _);

                    vkCmdSetDepthBiasEnable(cmd, false as _);

                    vkCmdSetStencilTestEnable(cmd, false as _);

                    let vkCmdSetPolygonModeEXT = unsafe {
                        transmute::<_, vkCmdSetPolygonModeEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_POLYGON_MODE_EXT,
                        ))
                    };
                    unsafe { (vkCmdSetPolygonModeEXT)(cmd, PolygonMode::Fill) };

                    let vkCmdSetRasterizationSamplesEXT = unsafe {
                        transmute::<_, vkCmdSetRasterizationSamplesEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_RASTERIZATION_SAMPLES_EXT,
                        ))
                    };
                    unsafe {
                        (vkCmdSetRasterizationSamplesEXT)(cmd, SampleCountFlagBits::_1 as u32)
                    };

                    let vkCmdSetSampleMaskEXT = unsafe {
                        transmute::<_, vkCmdSetSampleMaskEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_SAMPLE_MASK_EXT,
                        ))
                    };
                    unsafe {
                        (vkCmdSetSampleMaskEXT)(
                            cmd,
                            SampleCountFlagBits::_1 as u32,
                            &SampleMask(u64::MAX as *const _),
                        )
                    };

                    let vkCmdSetAlphaToCoverageEnableEXT = unsafe {
                        transmute::<_, vkCmdSetAlphaToCoverageEnableEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_ALPHA_TO_COVERAGE_ENABLE_EXT,
                        ))
                    };
                    unsafe { (vkCmdSetAlphaToCoverageEnableEXT)(cmd, false as _) };

                    let vkCmdSetColorBlendEnableEXT = unsafe {
                        transmute::<_, vkCmdSetColorBlendEnableEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_COLOR_BLEND_ENABLE_EXT,
                        ))
                    };
                    unsafe { (vkCmdSetColorBlendEnableEXT)(cmd, 0, 1, [false as _].as_ptr()) };

                    let vkCmdSetColorBlendEquationEXT = unsafe {
                        transmute::<_, vkCmdSetColorBlendEquationEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_COLOR_BLEND_EQUATION_EXT,
                        ))
                    };
                    unsafe {
                        (vkCmdSetColorBlendEquationEXT)(
                            cmd,
                            0,
                            1,
                            [ColorBlendEquationEXT {
                                src_color_blend_factor: BlendFactor::One,
                                dst_color_blend_factor: BlendFactor::Zero,
                                color_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::Zero,
                                alpha_blend_op: BlendOp::Add,
                            }]
                            .as_ptr(),
                        )
                    };

                    let vkCmdSetColorWriteMaskEXT = unsafe {
                        transmute::<_, vkCmdSetColorWriteMaskEXT>(load_device_fn(
                            device,
                            VK_CMD_SET_COLOR_WRITE_MASK_EXT,
                        ))
                    };
                    unsafe {
                        (vkCmdSetColorWriteMaskEXT)(
                            cmd,
                            0,
                            1,
                            [ColorComponentFlagBits::R as u32
                                | ColorComponentFlagBits::G as u32
                                | ColorComponentFlagBits::B as u32
                                | ColorComponentFlagBits::A as u32]
                            .as_ptr(),
                        )
                    };
                }
            }

            pub(crate) fn draw(cmd: CommandBuffer) {
                unsafe { vkCmdDraw(cmd, 3, 1, 0, 0) };
            }
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
                        1920,
                        1080,
                        1,
                        XBlackPixel(display, screen),
                        XWhitePixel(display, screen),
                    );
                    XMapWindow(display, window);

                    (display, window)
                };
                Self { display, window }
            }

            fn update(&self) {
                let mut buf = crate::core_alloc::vec![0u8; 65535];
                unsafe {
                    XNextEvent(self.display, buf.as_mut_ptr() as *mut _);
                }
            }

            fn size(&self) -> (u32, u32) {
                unsafe {
                    let screen = XScreenOfDisplay(self.display, 0);
                    (
                        XWidthOfScreen(screen) as u32,
                        XHeightOfScreen(screen) as u32,
                    )
                }
            }

            fn handle(&self) -> SurfaceHandle {
                SurfaceHandle::Linux {
                    display: self.display,
                    window: self.window,
                }
            }
        }

        extern "C" {
            fn XScreenOfDisplay(display: *const c_void, num: u32) -> *const c_void;
            fn XWidthOfScreen(screen: *const c_void) -> i32;
            fn XHeightOfScreen(screen: *const c_void) -> i32;
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

    pub fn make_platform_renderer() -> (Box<dyn Renderer>, Box<dyn Surface>) {
        let surface = make_platform_surface();
        #[cfg(target_os = "linux")]
        (Box::new(vk::Renderer::create(&*surface)), surface)
    }

    pub enum SurfaceHandle {
        Linux {
            window: *const c_void,
            display: *const c_void,
        },
    }

    pub trait Surface {
        fn open() -> Self
        where
            Self: Sized;
        fn update(&self);
        fn size(&self) -> (u32, u32);
        fn handle(&self) -> SurfaceHandle;
    }

    pub trait Renderer {
        fn create(surface: &dyn Surface) -> Self
        where
            Self: Sized;
        fn render(&mut self);
        fn camera(&mut self) -> &mut Camera;
    }

    #[derive(Default, Clone, Copy)]
    pub struct Camera {
        translation: Vector<3>,
        rotation: Quaternion,
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

pub mod math {
    use core::{iter, mem};
    use core::ops::Neg;

    pub struct Projection {}

    pub trait Math {
        fn add(self, rhs: Self) -> Self;
        fn mul(self, rhs: Self) -> Self;
    }

    pub trait Number: Math + Copy {}

    pub trait Negate {
        fn negate(self) -> Self;
    }

    impl Math for f32 {
        fn add(self, rhs: Self) -> Self {
            self + rhs
        }

        fn mul(self, rhs: Self) -> Self {
            self * rhs
        }
    }

    impl Negate for f32 {
        fn negate(self) -> Self {
            -self
        }
    }

    pub trait Inverse {
        fn inverse(self) -> Self;
    }

    impl Inverse for f32 {
        fn inverse(self) -> Self {
            1.0 / self
        }
    }

    impl Number for f32 {}

    pub trait VectorMath<const N: usize, T: Number>: Math {
        fn scale(self, amount: T) -> Self;
    }

    impl<T: Number, const N: usize> Math for [T; N] {
        fn add(self, rhs: Self) -> Self {
            let mut data = unsafe { mem::zeroed::<Self>() };
            for (i, x) in self.into_iter().zip(rhs).map(|(x, y)| x.add(y)).enumerate() {
                data[i] = x;
            }
            data
        }

        fn mul(self, rhs: Self) -> Self {
            let mut data = unsafe { mem::zeroed::<Self>() };
            for (i, x) in self.into_iter().zip(rhs).map(|(x, y)| x.mul(y)).enumerate() {
                data[i] = x;
            }
            data
        }
    }

    impl<T: Number + Negate, const N: usize> Negate for [T; N] {
        fn negate(mut self) -> Self {
            for x in &mut self {
                *x = x.negate();
            }
            self
        }
    }

    impl<T: Number + Inverse, const N: usize> Inverse for [T; N] {
        fn inverse(mut self) -> Self {
            for x in &mut self {
                *x = x.inverse();
            }
            self
        }
    }

    impl<const N: usize, T: Number> VectorMath<N, T> for [T; N] {
        fn scale(mut self, amount: T) -> Self {
            for x in &mut self {
                *x = x.mul(amount);
            }
            self
        }
    }

    #[derive(Clone, Copy)]
    pub struct Vector<const N: usize, T = f32>([T; N]);

    impl<const N: usize, T: Default + Copy> Default for Vector<N, T> {
        fn default() -> Self {
            Self([T::default(); N])
        }
    }

    impl<const N: usize, T: Number> core::ops::Add for Vector<N, T> {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0.add(rhs.0))
        }
    }

    impl<const N: usize, T: Number> core::ops::AddAssign for Vector<N, T> {
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }

    impl<const N: usize, T: Number + Negate> core::ops::Sub for Vector<N, T> {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0.add(rhs.0.negate()))
        }
    }

    impl<const N: usize, T: Number + Negate> core::ops::SubAssign for Vector<N, T> {
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }

    impl<const N: usize, T: Number> core::ops::Mul for Vector<N, T> {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            Self(self.0.mul(rhs.0))
        }
    }

    impl<const N: usize, T: Number> core::ops::MulAssign for Vector<N, T> {
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }

    impl<const N: usize, T: Number + Inverse> core::ops::Div for Vector<N, T> {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            Self(self.0.mul(rhs.0.inverse()))
        }
    }

    impl<const N: usize, T: Number + Inverse> core::ops::DivAssign for Vector<N, T> {
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }

    impl MatrixMath<4, 4> for [f32; 16] {}

    pub trait MatrixMath<const N: usize, const M: usize> {}

    pub struct Matrix<const N: usize, const M: usize, T = f32>([T; N * M])
    where
        [T; N * M]: MatrixMath<N, M>;

    #[derive(Clone, Copy, Default)]
    pub struct Quaternion {
        imaginary: Vector<3>,
        real: f32,
    }

    impl Quaternion {
        fn to_matrix(self) -> Matrix<4, 4> {
            let Self {
                real: w,
                imaginary: Vector([x, y, z]),
            } = self;
            let _x = [
                w * w + x * x - y * y - z * z,
                2. * x * y + 2. * w * z,
                2. * x * z - 2. * w * y,
                0.,
            ];
            let _y = [
                2. * x * y - 2. * w * z,
                w * w - x * x + y * y - z * z,
                2. * y * z - 2. * w * x,
                0.,
            ];
            let _z = [
                2. * x * z + 2. * w * y,
                2. * y * z + 2. * w * x,
                w * w - x * x - y * y + z * z,
                0.,
            ];
            let _w = [0., 0., 0., 1.];

            let mut m = [0.; 16];
            iter::chain(_x, _y)
                .chain(_z)
                .chain(_w)
                .enumerate()
                .for_each(|(i, x)| m[i] = x);
            Matrix(m)
        }
    }
}
