use core::str::FromStr;

use crate::prelude::*;

use super::include::*;

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
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub flags: RenderingFlags,
    pub render_area: Rect2D,
    pub layer_count: u32,
    pub view_mask: u32,
    pub color_attachment_count: u32,
    pub p_color_attachments: *const RenderingAttachmentInfo,
    pub p_depth_attachment: *const RenderingAttachmentInfo,
    pub p_stencil_attachment: *const RenderingAttachmentInfo,
}

#[repr(C)]
pub struct Rect2D {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

#[repr(C)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
pub struct RenderingAttachmentInfo {
    pub s_type: StructureType,
    pub p_next: *const c_void,
    pub image_view: ImageView,
    pub image_layout: ImageLayout,
    pub resolve_mode: ResolveMode,
    pub resolve_image_view: ImageView,
    pub resolve_image_layout: ImageLayout,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub clear_value: ClearValue,
}

pub type RenderingFlags = u32;

#[repr(u32)]
pub enum RenderingFlagBits {
    ContentsSecondaryCommandBuffers = 0x00000001,
    Suspending = 0x00000002,
    Resuming = 0x00000004,
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
    pub color: ClearColorValue,
    pub depth_stencil: ClearDepthStencilValue,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ClearColorValue {
    pub float32: [f32; 4],
    pub int32: [i32; 4],
    pub uint32: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClearDepthStencilValue {
    depth: f32,
    stencil: u32,
}

pub struct RenderAttachment {
    pub image_view: ImageView,
    pub image_layout: ImageLayout,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub clear_value: ClearValue,
}

pub struct Render {
    pub width: u32,
    pub height: u32,
    pub color: Vec<RenderAttachment>,
    pub depth: Option<RenderAttachment>,
    pub stencil: Option<RenderAttachment>,
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

pub type SampleMask = u64;

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
#[derive(Debug, Clone, Copy)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
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
                dbg!(e);
                Err(Error::Unknown)
            }
        }
    }
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
    pub src_offset: DeviceSize,
    pub dst_offset: DeviceSize,
    pub size: DeviceSize,
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

#[derive(Clone, Copy)]
pub struct DeviceAddress(u64);

#[derive(Clone, Copy)]
pub struct Push {
    pub global_buffer: DeviceAddress,
}

pub struct PushConstant<T: Copy> {
    pub offset: u32,
    pub data: T,
}

pub fn resolution(physical_device: PhysicalDevice, surface: Surface) -> (u32, u32) {
    let mut info = unsafe { mem::zeroed::<SurfaceCapabilitiesKHR>() };
    unsafe { vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &mut info) };
    (info.current_extent.width, info.current_extent.height)
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

pub unsafe fn load_device_fn(device: Device, name: &str) -> fn() {
    let compile_name = CString::from_str(name).unwrap();
    vkGetDeviceProcAddr(device, compile_name.as_ref().as_ptr())
}

#[derive(Debug)]
pub enum Error {
    Unknown,
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

pub fn c_str_array(
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
    MemoryAllocateInfo = 5,
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
