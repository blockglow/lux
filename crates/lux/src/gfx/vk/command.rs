use core::ptr;

use crate::prelude::*;

use super::include::*;

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

#[derive(Clone, Copy, Debug)]
pub struct BufferBarrier {
    pub src_access: AccessFlags,
    pub dst_access: AccessFlags,
    pub buffer: Buffer,
    pub offset: DeviceSize,
    pub size: DeviceSize,
}

pub type CommandPoolCreateFlags = u32;

#[repr(C)]
pub struct CommandPoolCreateInfo {
    pub s_type: StructureType,
    pub p_next: *const core::ffi::c_void,
    pub flags: CommandPoolCreateFlags,
    pub queue_family_index: QueueFamilyIndex,
}

#[derive(Clone, Copy)]
pub struct CommandBuffer(*const c_void);

#[derive(Resource)]
pub struct CommandBuffers(Vec<CommandBuffer>);

impl ops::Deref for CommandBuffers {
    type Target = [CommandBuffer];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl Default for CommandBuffer {
    fn default() -> Self {
        Self(ptr::null())
    }
}

impl CommandBuffer {
    pub(crate) fn allocate(
        device: ResMut<Device>,
        command_pool: ResMut<CommandPool>,
    ) -> Insert<CommandBuffers> {
        let alloc_info = CommandBufferAllocateInfo {
            s_type: StructureType::CommandBufferAllocateInfo,
            p_next: ptr::null(),
            command_pool: *command_pool,
            level: CommandBufferLevel::Primary,
            command_buffer_count: 4,
        };

        let mut command_buffers = vec![CommandBuffer::default(); 4];
        VkResult::handle(unsafe {
            vkAllocateCommandBuffers(*device, &alloc_info, command_buffers.as_mut_ptr())
        })
        .unwrap();

        CommandBuffers(command_buffers).into()
    }

    pub(crate) fn begin(self) {
        let info = CommandBufferBeginInfo {
            s_type: StructureType::CommandBufferBeginInfo,
            p_next: ptr::null(),
            flags: 0,
            p_inheritance_info: None,
        };
        unsafe { vkBeginCommandBuffer(self, &info) };
    }

    pub(crate) fn end(self) {
        unsafe { vkEndCommandBuffer(self) };
    }

    pub(crate) fn begin_render(self, render: Render) {
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
            p_depth_attachment: depth.as_ref().map(|x| x as *const _).unwrap_or(ptr::null()),
            p_stencil_attachment: stencil
                .as_ref()
                .map(|x| x as *const _)
                .unwrap_or(ptr::null()),
        };
        unsafe { vkCmdBeginRendering(self, &info) };
    }

    pub fn copy_buffer(
        self,
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
        unsafe { vkCmdCopyBuffer(self, src, dst, 1, &copy) }
    }

    pub(crate) fn end_render(self) {
        unsafe { vkCmdEndRendering(self) };
    }

    pub fn push_constant<T: Copy>(self, layout: PipelineLayout, push: PushConstant<T>) {
        unsafe {
            vkCmdPushConstants(
                self,
                layout,
                ShaderStageFlagBits::All as u32,
                push.offset,
                mem::size_of::<T>() as u32,
                &push.data as *const _ as *const _,
            )
        };
    }

    pub fn pipeline_barrier_buffer(
        self,
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
                self,
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
        self,
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
                self,
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
        self,
        device: Device,
        stages: Vec<ShaderStageFlags>,
        shaders: Vec<Shader>,
    ) {
        let vkCmdBindShadersEXT = unsafe {
            mem::transmute::<_, vkCmdBindShadersEXT>(load_device_fn(
                device,
                VK_CMD_BIND_SHADERS_EXT,
            ))
        };
        unsafe {
            (vkCmdBindShadersEXT)(
                self,
                shaders.len() as u32,
                stages.as_ptr(),
                shaders.as_ptr(),
            )
        };
    }

    pub(crate) fn draw_settings(self, device: Device, width: u32, height: u32) {
        unsafe {
            vkCmdSetViewportWithCount(
                self,
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
                self,
                1,
                &Rect2D {
                    offset: Offset2D { x: 0, y: 0 },
                    extent: Extent2D { width, height },
                },
            );
            vkCmdSetPrimitiveTopology(self, PrimitiveTopology::TriangleList);
            vkCmdSetRasterizerDiscardEnable(self, false as _);
            vkCmdSetPrimitiveRestartEnable(self, false as _);
            vkCmdSetCullMode(self, CullModeFlagBits::None as u32);

            vkCmdSetFrontFace(self, FrontFace::CounterClockwise);
            vkCmdSetDepthTestEnable(self, false as _);

            vkCmdSetDepthWriteEnable(self, false as _);

            vkCmdSetDepthBiasEnable(self, false as _);

            vkCmdSetStencilTestEnable(self, false as _);

            let vkCmdSetPolygonModeEXT = unsafe {
                mem::transmute::<_, vkCmdSetPolygonModeEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_POLYGON_MODE_EXT,
                ))
            };
            unsafe { (vkCmdSetPolygonModeEXT)(self, PolygonMode::Fill) };

            let vkCmdSetRasterizationSamplesEXT = unsafe {
                mem::transmute::<_, vkCmdSetRasterizationSamplesEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_RASTERIZATION_SAMPLES_EXT,
                ))
            };
            unsafe { (vkCmdSetRasterizationSamplesEXT)(self, SampleCountFlagBits::_1 as u32) };

            let vkCmdSetSampleMaskEXT = unsafe {
                mem::transmute::<_, vkCmdSetSampleMaskEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_SAMPLE_MASK_EXT,
                ))
            };
            unsafe {
                (vkCmdSetSampleMaskEXT)(
                    self,
                    SampleCountFlagBits::_1 as u32,
                    &u64::MAX as *const _ as *const _,
                )
            };

            let vkCmdSetAlphaToCoverageEnableEXT = unsafe {
                mem::transmute::<_, vkCmdSetAlphaToCoverageEnableEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_ALPHA_TO_COVERAGE_ENABLE_EXT,
                ))
            };
            unsafe { (vkCmdSetAlphaToCoverageEnableEXT)(self, false as _) };

            let vkCmdSetColorBlendEnableEXT = unsafe {
                mem::transmute::<_, vkCmdSetColorBlendEnableEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_COLOR_BLEND_ENABLE_EXT,
                ))
            };
            unsafe { (vkCmdSetColorBlendEnableEXT)(self, 0, 1, [false as _].as_ptr()) };

            let vkCmdSetColorBlendEquationEXT = unsafe {
                mem::transmute::<_, vkCmdSetColorBlendEquationEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_COLOR_BLEND_EQUATION_EXT,
                ))
            };
            unsafe {
                (vkCmdSetColorBlendEquationEXT)(
                    self,
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
                mem::transmute::<_, vkCmdSetColorWriteMaskEXT>(load_device_fn(
                    device,
                    VK_CMD_SET_COLOR_WRITE_MASK_EXT,
                ))
            };
            unsafe {
                (vkCmdSetColorWriteMaskEXT)(
                    self,
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

    pub(crate) fn draw(self) {
        unsafe { vkCmdDraw(self, 6, 1, 0, 0) };
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CommandPoolCreateFlagBits {
    Transient = 0x00000001,
    ResetCommandBuffer = 0x00000002,
    Protected = 0x00000004,
}

#[derive(Clone, Copy, Resource)]
pub struct CommandPool(*const c_void);

impl Default for CommandPool {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl CommandPool {
    pub(crate) fn new(
        device: ResMut<Device>,
        queue_family_index: ResMut<QueueFamilyIndex>,
    ) -> Insert<Self> {
        let command_pool_info = CommandPoolCreateInfo {
            s_type: StructureType::CommandPoolCreateInfo,
            p_next: ptr::null(),
            flags: CommandPoolCreateFlagBits::ResetCommandBuffer as u32,
            queue_family_index: *queue_family_index,
        };

        let mut command_pool = default();
        VkResult::handle(unsafe {
            vkCreateCommandPool(*device, &command_pool_info, ptr::null(), &mut command_pool)
        })
        .unwrap();

        command_pool.into()
    }
}

impl Buffer {
    pub(crate) fn create(device: Device, size: DeviceSize, usage: BufferUsageFlags) -> Self {
        let info = BufferCreateInfo {
            s_type: StructureType::BufferCreateInfo,
            p_next: ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: SharingMode::Exclusive,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut buffer = Self::default();
        unsafe { vkCreateBuffer(device, &info, ptr::null(), &mut buffer) };
        buffer
    }

    pub(crate) fn requirements(self, device: Device) -> MemoryRequirements {
        let mut req = unsafe { mem::zeroed::<MemoryRequirements>() };
        unsafe { vkGetBufferMemoryRequirements(device, self, &mut req) };
        req
    }

    pub(crate) fn device_address(self, device: Device) -> DeviceAddress {
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
pub struct CommandBufferBeginInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: CommandBufferUsageFlags,
    pub p_inheritance_info: Option<*const c_void>,
}

pub type CommandBufferUsageFlags = u32;

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CommandBufferUsageFlagBits {
    OneTimeSubmit = 0x00000001,
    RenderPassContinue = 0x00000002,
    SimultaneousUse = 0x00000004,
}
