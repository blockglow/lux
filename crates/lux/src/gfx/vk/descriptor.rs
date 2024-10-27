use core::ops::Deref;

use crate::prelude::*;

use super::include::*;

#[derive(Resource)]
pub struct DescriptorTable(BTreeMap<DescriptorType, u32>);

impl Deref for DescriptorTable {
    type Target = BTreeMap<DescriptorType, u32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DescriptorTable {
    pub(crate) fn new() -> Insert<Self> {
        Self::default().into()
    }
}

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

#[derive(Resource)]
pub struct DescriptorSets(Vec<DescriptorSet>);

impl DescriptorSet {
    pub(crate) fn allocate(
        device: ResMut<Device>,
        descriptor_pool: ResMut<DescriptorPool>,
        descriptor_set_layout: ResMut<DescriptorSetLayout>,
    ) -> Insert<DescriptorSets> {
        let set_layouts = vec![*descriptor_set_layout; 4];
        let alloc_info = DescriptorSetAllocateInfo {
            s_type: StructureType::DescriptorSetAllocateInfo,
            p_next: ptr::null(),
            descriptor_pool: *descriptor_pool,
            descriptor_set_count: 4,
            p_set_layouts: set_layouts.as_ptr(),
        };

        let mut descriptor_sets = vec![DescriptorSet::default(); 4];
        VkResult::handle(unsafe {
            vkAllocateDescriptorSets(*device, &alloc_info, descriptor_sets.as_mut_ptr())
        })
        .unwrap();

        DescriptorSets(descriptor_sets).into()
    }
}

#[derive(Clone, Copy, Resource)]
pub struct DescriptorPool(*const c_void);

impl Default for DescriptorPool {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl DescriptorPool {
    pub(crate) fn bindless(device: ResMut<Device>, table: Res<DescriptorTable>) -> Insert<Self> {
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
                *device,
                &descriptor_pool_info,
                ptr::null(),
                &mut descriptor_pool,
            )
        })
        .unwrap();
        descriptor_pool.into()
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
#[derive(Clone, Copy, Resource)]
pub struct DescriptorSetLayout(*const c_void);

impl Default for DescriptorSetLayout {
    fn default() -> Self {
        Self(ptr::null())
    }
}
impl DescriptorSetLayout {
    pub(crate) fn bindless(device: ResMut<Device>, table: ResMut<DescriptorTable>) -> Insert<Self> {
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
                *device,
                &descriptor_set_layout_info,
                ptr::null(),
                &mut descriptor_set_layout,
            )
        })
        .unwrap();
        descriptor_set_layout.into()
    }
}
