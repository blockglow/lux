use core::str::FromStr;

use crate::prelude::*;

use super::include::*;

pub struct ShaderSource(pub Vec<u8>);

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
    pub stage: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
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
    pub flags: ShaderCreateFlags,
    pub source: ShaderSource,
    pub stage: ShaderStageFlags,
    pub next_stage: ShaderStageFlags,
    pub push_constant_range: PushConstantRange,
    pub entry: String,
}

#[derive(Clone, Copy)]
pub struct Shader(*const c_void);

impl Default for Shader {
    fn default() -> Self {
        Self(ptr::null())
    }
}

impl Shader {
    pub(crate) fn compile(
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
            mem::transmute::<_, vkCreateShadersEXT>(load_device_fn(device, VK_CREATE_SHADERS_EXT))
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
