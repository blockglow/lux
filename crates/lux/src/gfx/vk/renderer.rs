use core::iter;

use crate::gfx::{Camera, Renderer as GfxRenderer, Surface as GfxSurface};
use crate::gfx::vk::instance::Instance;
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
            DescriptorSet::allocate(device, descriptor_pool, descriptor_set_layout).unwrap();
        let vertex = include_bytes!("../../../../../assets/shader/vertex.spirv").to_vec();
        let fragment = include_bytes!("../../../../../assets/shader/fragment.spirv").to_vec();
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
        let shaders = Shader::compile(device, &[descriptor_set_layout], shader_specs).unwrap();
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
            BufferUsageFlagBits::StorageBuffer as u32 | BufferUsageFlagBits::TransferDst as u32,
        );

        let memory = {
            let req = global_buffer.requirements(device);
            Memory::device_local(physical_device, device, req).bind_buffer(device, global_buffer)
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
            self.camera
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

        let cmd = self.command_buffers[idx as usize];
        cmd.begin();

        cmd.pipeline_barrier_buffer(
            PipelineStageFlagBits::TopOfPipe as u32,
            PipelineStageFlagBits::Transfer as u32,
            &[BufferBarrier {
                src_access: 0,
                dst_access: AccessFlagBits::TransferWrite as u32,
                buffer: self.global_buffer,
                offset: 0,
                size: 64,
            }],
        );

        self.staging_buffer.device_copy(cmd);

        cmd.pipeline_barrier_buffer(
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

        cmd.pipeline_barrier_image(
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
        cmd.begin_render(
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

        cmd.push_constant(
            self.pipeline_layout,
            PushConstant {
                offset: 0,
                data: Push {
                    global_buffer: self.global_buffer.device_address(self.device),
                },
            },
        );
        cmd.bind_shaders(
            self.device,
            vec![
                ShaderStageFlagBits::Vertex as u32,
                ShaderStageFlagBits::Fragment as u32,
            ],
            self.shaders.clone(),
        );
        cmd.draw_settings(self.device, width, height);
        cmd.draw();
        cmd.end_render();
        cmd.pipeline_barrier_image(
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

        cmd.end();

        self.queue.submit(
            self.image_avail_semaphore[self.frame],
            self.render_finish_semaphore[self.frame],
            self.in_flight_fence[self.frame],
            cmd
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

    fn set_voxels(&mut self, id: u64, voxels: Box<dyn Volume>) {
        todo!()
    }
}
