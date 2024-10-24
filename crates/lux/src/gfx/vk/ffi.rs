use super::include::*;

pub type vkCmdBindShadersEXT = fn(
    cmd: CommandBuffer,
    stage_count: u32,
    stages: *const ShaderStageFlags,
    shaders: *const Shader,
);
pub const VK_CMD_BIND_SHADERS_EXT: &str = "vkCmdBindShadersEXT";

pub type vkCreateShadersEXT = fn(
    device: Device,
    count: u32,
    infos: *const ShaderCreateInfo,
    alloc: *const c_void,
    shaders: *mut Shader,
) -> VkResult;
pub const VK_CREATE_SHADERS_EXT: &str = "vkCreateShadersEXT";

pub type vkCmdSetPolygonModeEXT = fn(command_buffer: CommandBuffer, polygon_mode: PolygonMode);
pub const VK_CMD_SET_POLYGON_MODE_EXT: &str = "vkCmdSetPolygonModeEXT";

pub type vkCmdSetRasterizationSamplesEXT =
    fn(command_buffer: CommandBuffer, rasterization_samples: SampleCountFlags);
pub const VK_CMD_SET_RASTERIZATION_SAMPLES_EXT: &str = "vkCmdSetRasterizationSamplesEXT";

pub type vkCmdSetSampleMaskEXT =
    fn(command_buffer: CommandBuffer, samples: SampleCountFlags, p_sample_mask: *const SampleMask);
pub const VK_CMD_SET_SAMPLE_MASK_EXT: &str = "vkCmdSetSampleMaskEXT";

pub type vkCmdSetAlphaToCoverageEnableEXT =
    fn(command_buffer: CommandBuffer, alpha_to_coverage_enable: Bool32);
pub const VK_CMD_SET_ALPHA_TO_COVERAGE_ENABLE_EXT: &str = "vkCmdSetAlphaToCoverageEnableEXT";

pub type vkCmdSetColorBlendEnableEXT = fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_blend_enables: *const Bool32,
);
pub const VK_CMD_SET_COLOR_BLEND_ENABLE_EXT: &str = "vkCmdSetColorBlendEnableEXT";

pub type vkCmdSetColorBlendEquationEXT = fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_blend_equations: *const ColorBlendEquationEXT,
);
pub const VK_CMD_SET_COLOR_BLEND_EQUATION_EXT: &str = "vkCmdSetColorBlendEquationEXT";

pub type vkCmdSetColorWriteMaskEXT = fn(
    command_buffer: CommandBuffer,
    first_attachment: u32,
    attachment_count: u32,
    p_color_write_masks: *const ColorComponentFlags,
);
pub const VK_CMD_SET_COLOR_WRITE_MASK_EXT: &str = "vkCmdSetColorWriteMaskEXT";

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
    pub fn vkCmdPipelineBarrier(
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

    pub fn vkCmdSetDepthTestEnable(command_buffer: CommandBuffer, depth_test_enable: Bool32);

    pub fn vkCmdSetDepthWriteEnable(command_buffer: CommandBuffer, depth_write_enable: Bool32);

    pub fn vkCmdSetDepthBiasEnable(command_buffer: CommandBuffer, depth_bias_enable: Bool32);

    pub fn vkCmdSetStencilTestEnable(command_buffer: CommandBuffer, stencil_test_enable: Bool32);

    pub fn vkQueueSubmit(
        queue: Queue,
        submit_count: u32,
        submits: *const SubmitInfo,
        fence: Fence,
    ) -> VkResult;
    pub fn vkQueuePresentKHR(queue: Queue, present: *const PresentInfo) -> VkResult;
}
