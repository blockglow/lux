use crate::gfx::vk::instance::Instance;
use crate::gfx::{Camera, Renderer as GfxRenderer, Surface as GfxSurface};
use crate::prelude::*;
use crate::voxel::Volume;

use super::include::*;

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
        // let instance = Instance::new().unwrap();
        // let physical_device = PhysicalDevice::acquire_best(instance).unwrap();
        // let surface = Surface::open(instance, surface).unwrap();
        // let (device, qfi) = Device::new(physical_device, surface).unwrap();
        // let queue = qfi.queue(device);
        // let resolution = resolution(physical_device, surface);
        // let (swapchain, swapchain_images) =
        //     Swapchain::new(device, qfi, surface, None, resolution).unwrap();
        // let swapchain_image_views = swapchain_images
        //     .iter()
        //     .copied()
        //     .map(|image| {
        //         ImageView::create(device, image, ImageViewType::Type2d, Format::Bgra8Unorm)
        //     })
        //     .try_collect::<Vec<_>>()
        //     .unwrap();
        // let descriptor_table = DescriptorTable::default();
        // let descriptor_pool = DescriptorPool::bindless(device, &descriptor_table).unwrap();
        // let descriptor_set_layout =
        //     DescriptorSetLayout::bindless(device, &descriptor_table).unwrap();
        // let descriptor_sets =
        //     DescriptorSet::allocate(device, descriptor_pool, descriptor_set_layout).unwrap();
        // let vertex = include_bytes!("../../../../../assets/shader/vertex.spirv").to_vec();
        // let fragment = include_bytes!("../../../../../assets/shader/fragment.spirv").to_vec();
        // let shader_specs = vec![
        //     ShaderSpec {
        //         flags: 0,
        //         source: ShaderSource(vertex),
        //         stage: ShaderStageFlagBits::Vertex as u32,
        //         next_stage: ShaderStageFlagBits::Fragment as u32,
        //         push_constant_range: PushConstantRange {
        //             stage: ShaderStageFlagBits::All as u32,
        //             offset: 0,
        //             size: 64,
        //         },
        //         entry: "main".to_string(),
        //     },
        //     ShaderSpec {
        //         flags: 0,
        //         source: ShaderSource(fragment),
        //         stage: ShaderStageFlagBits::Fragment as u32,
        //         next_stage: 0,
        //         push_constant_range: PushConstantRange {
        //             stage: ShaderStageFlagBits::All as u32,
        //             offset: 0,
        //             size: 64,
        //         },
        //         entry: "main".to_string(),
        //     },
        // ];
        // let shaders = Shader::compile(device, &[descriptor_set_layout], shader_specs).unwrap();
        // let command_pool = CommandPool::new(device, qfi).unwrap();
        // let command_buffers = CommandBuffer::allocate(device, command_pool, 3).unwrap();
        // let in_flight_fence = iter::repeat_with(|| Fence::create_signaled(device))
        //     .take(3)
        //     .try_collect::<Vec<_>>()
        //     .unwrap();
        // let image_avail_semaphore = iter::repeat_with(|| Semaphore::create(device))
        //     .take(3)
        //     .try_collect::<Vec<_>>()
        //     .unwrap();
        // let render_finish_semaphore = iter::repeat_with(|| Semaphore::create(device))
        //     .take(3)
        //     .try_collect::<Vec<_>>()
        //     .unwrap();
        //
        // let global_buffer = Buffer::create(
        //     device,
        //     1024,
        //     BufferUsageFlagBits::StorageBuffer as u32 | BufferUsageFlagBits::TransferDst as u32,
        // );
        //
        // let memory = {
        //     let req = global_buffer.requirements(device);
        //     Memory::device_local(physical_device, device, req).bind_buffer(device, global_buffer)
        // };
        //
        // let staging_buffer = StagingBuffer::create(physical_device, device, 1_000_000);
        //
        // let pipeline_layout = PipelineLayout::new(device, descriptor_set_layout);
        //
        // Self {
        //     instance,
        //     physical_device,
        //     device,
        //     qfi,
        //     surface,
        //     swapchain,
        //     swapchain_images,
        //     swapchain_image_views,
        //     descriptor_table,
        //     descriptor_pool,
        //     descriptor_set_layout,
        //     descriptor_sets,
        //     shaders,
        //     command_pool,
        //     command_buffers,
        //     in_flight_fence,
        //     image_avail_semaphore,
        //     render_finish_semaphore,
        //     queue,
        //     global_buffer,
        //     pipeline_layout,
        //     staging_buffer,
        //     frame: 0,
        //     camera: Camera::default(),
        // }
        todo!()
    }

    fn render(&mut self) {}

    fn camera(&mut self) -> &mut Camera {
        &mut self.camera
    }

    fn set_voxels(&mut self, id: u64, voxels: Box<dyn Volume>) {
        todo!()
    }
}
