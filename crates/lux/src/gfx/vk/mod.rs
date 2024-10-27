pub use ecs::Vulkan;

pub use crate::ecs::*;

pub mod include {
    pub use crate::prelude::*;

    pub use super::buffer::*;
    pub use super::command::*;
    pub use super::descriptor::*;
    pub use super::device::*;
    pub use super::ffi::*;
    pub use super::image::*;
    pub use super::instance::*;
    pub use super::memory::*;
    pub use super::physical_device::*;
    pub use super::pipeline_layout::*;
    pub use super::qfi::*;
    pub use super::renderer2::*;
    pub use super::shader::*;
    pub use super::staging::*;
    pub use super::surface::*;
    pub use super::swapchain::*;
    pub use super::sync::*;
    pub use super::util::*;
}

mod buffer;
mod command;
mod descriptor;
mod device;
mod ffi;
mod image;
mod instance;
mod memory;
mod physical_device;
mod pipeline_layout;
mod qfi;
pub mod renderer2;
mod shader;
mod staging;
mod surface;
mod swapchain;
mod sync;
mod util;

mod ecs {
    use std::mem::swap;
    use std::process::Command;

    use crate::gfx::{ActiveSurface, Camera, PlatformSurface};

    use super::include::*;

    #[derive(Resource)]
    pub struct Vulkan {}

    impl Plugin for Vulkan {
        fn add(self, app: App) -> App {
            app.insert(self)
                .insert(FrameIndex::default())
                .insert(Camera::default())
                .schedule(
                    (
                        Instance::new,
                        Surface::open,
                        PhysicalDevice::acquire_best,
                        Device::new,
                        Swapchain::new,
                        Swapchain::images,
                        DescriptorTable::new,
                        DescriptorPool::bindless,
                        DescriptorSetLayout::bindless,
                        DescriptorSet::allocate,
                        StandardShaders::compile,
                        CommandPool::new,
                        CommandBuffer::allocate,
                        FrameSync::new,
                        PipelineLayout::new,
                        StagingBuffer::create,
                        GlobalBuffer::new,
                    )
                        .chain()
                        .run_if(stage(Start)),
                )
                .schedule(
                    (
                        Vulkan::wait_and_reset,
                        Vulkan::write,
                        Vulkan::swapchain,
                        Vulkan::render,
                        Vulkan::submit_and_present,
                    )
                        .chain()
                        .run_if(stage(Update)),
                )
        }
    }

    impl Vulkan {
        fn wait_and_reset(
            device: ResMut<Device>,
            frame_sync: ResMut<FrameSync>,
            frame_index: Res<FrameIndex>,
        ) {
            if frame_sync[**frame_index].ready.wait(*device) {
                return;
            }

            frame_sync[**frame_index].ready.reset(*device);
        }

        fn write(
            mut staging_buffer: ResMut<StagingBuffer>,
            device: ResMut<Device>,
            global_buffer: ResMut<GlobalBuffer>,
            camera: ResMut<Camera>,
        ) {
            staging_buffer.write(**global_buffer, 0, *camera);
            staging_buffer.host_transfer(*device);
        }
        fn swapchain(
            mut swapchain: ResMut<Swapchain>,
            mut frame_images: ResMut<FrameImages>,
            physical_device: ResMut<PhysicalDevice>,
            device: ResMut<Device>,
            qfi: ResMut<QueueFamilyIndex>,
            surface: ResMut<Surface>,
            frame_sync: ResMut<FrameSync>,
            frame_index: ResMut<FrameIndex>,
        ) {
            let Ok(idx) = swapchain.next_image(*device, frame_sync[**frame_index].image_avail)
            else {
                let new_swapchain =
                    Swapchain::new(physical_device, device, qfi, surface, Some(swapchain));
                *swapchain = new_swapchain.0.unwrap();
                let swapchain_images = Swapchain::images(device, swapchain);
                *frame_images = swapchain_images.0.unwrap();
                return;
            };
        }

        fn render(
            physical_device: ResMut<PhysicalDevice>,
            surface: ResMut<Surface>,
            device: ResMut<Device>,
            cmds: ResMut<CommandBuffers>,
            frame_index: Res<FrameIndex>,
            global_buffer: ResMut<GlobalBuffer>,
            mut staging_buffer: ResMut<StagingBuffer>,
            frame_images: ResMut<FrameImages>,
            pipeline_layout: ResMut<PipelineLayout>,
            shaders: ResMut<StandardShaders>,
        ) {
            let Extent2D { width, height } = surface.capabilities(*physical_device).current_extent;
            let cmd = cmds[**frame_index as usize];
            cmd.begin();

            cmd.pipeline_barrier_buffer(
                PipelineStageFlagBits::TopOfPipe as u32,
                PipelineStageFlagBits::Transfer as u32,
                &[BufferBarrier {
                    src_access: 0,
                    dst_access: AccessFlagBits::TransferWrite as u32,
                    buffer: **global_buffer,
                    offset: 0,
                    size: 128,
                }],
            );

            staging_buffer.device_copy(cmd);

            cmd.pipeline_barrier_buffer(
                PipelineStageFlagBits::Transfer as u32,
                PipelineStageFlagBits::VertexShader as u32,
                &[BufferBarrier {
                    src_access: AccessFlagBits::TransferWrite as u32,
                    dst_access: AccessFlagBits::ShaderRead as u32,
                    buffer: **global_buffer,
                    offset: 0,
                    size: 128,
                }],
            );

            cmd.pipeline_barrier_image(
                0,
                PipelineStageFlagBits::TopOfPipe as u32,
                AccessFlagBits::ColorAttachmentWrite as u32,
                PipelineStageFlagBits::ColorAttachmentOutput as u32,
                vec![(
                    frame_images.images[**frame_index as usize],
                    ImageLayout::Undefined,
                    ImageLayout::ColorAttachmentOptimal,
                )],
            );
            cmd.begin_render(Render {
                width,
                height,
                color: vec![RenderAttachment {
                    image_view: frame_images.image_views[**frame_index as usize],
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
            });

            cmd.push_constant(
                *pipeline_layout,
                PushConstant {
                    offset: 0,
                    data: Push {
                        global_buffer: global_buffer.device_address(*device),
                    },
                },
            );
            cmd.bind_shaders(
                *device,
                vec![
                    ShaderStageFlagBits::Vertex as u32,
                    ShaderStageFlagBits::Fragment as u32,
                ],
                vec![shaders.vertex, shaders.fragment],
            );
            cmd.draw_settings(*device, width, height);
            cmd.draw();
            cmd.end_render();
            cmd.pipeline_barrier_image(
                AccessFlagBits::ColorAttachmentWrite as u32,
                PipelineStageFlagBits::ColorAttachmentOutput as u32,
                0,
                PipelineStageFlagBits::BottomOfPipe as u32,
                vec![(
                    frame_images.images[**frame_index as usize],
                    ImageLayout::ColorAttachmentOptimal,
                    ImageLayout::Present,
                )],
            );

            cmd.end();
        }

        fn submit_and_present(
            queue: ResMut<Queue>,
            cmds: ResMut<CommandBuffers>,
            swapchain: ResMut<Swapchain>,
            frame_sync: ResMut<FrameSync>,
            mut frame_index: ResMut<FrameIndex>,
        ) {
            let cmd = cmds[**frame_index as usize];
            queue.submit(
                frame_sync[**frame_index].image_avail,
                frame_sync[**frame_index].render_finish,
                frame_sync[**frame_index].ready,
                cmd,
            );
            queue.present(
                frame_sync[**frame_index].render_finish,
                *swapchain,
                **frame_index as u32,
            );

            **frame_index = (**frame_index + 1) % 3;
        }
    }
}
