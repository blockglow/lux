use super::include::*;

#[repr(C)]
#[derive(Clone, Copy, Resource)]
pub struct PipelineLayout(*const c_void);

impl Default for PipelineLayout {
    fn default() -> Self {
        Self(ptr::null())
    }
}

impl PipelineLayout {
    pub(crate) fn new(
        device: ResMut<Device>,
        descriptor_set_layout: ResMut<DescriptorSetLayout>,
    ) -> Insert<Self> {
        let info = PipelineLayoutCreateInfo {
            s_type: StructureType::PipelineLayoutCreateInfo,
            p_next: ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &*descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &PushConstantRange {
                stage: ShaderStageFlagBits::All as u32,
                offset: 0,
                size: 64,
            },
        };

        let mut layout = Self::default();
        unsafe { vkCreatePipelineLayout(*device, &info, ptr::null(), &mut layout) };
        layout.into()
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
