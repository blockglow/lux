use core::ffi::c_void;
use core::ptr;
use core::str::FromStr;

use crate::prelude::*;

use super::include::*;

#[repr(C)]
pub struct ApplicationInfo {
    s_type: StructureType,
    p_next: *const c_void,
    app_name: *const c_char,
    app_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
pub struct InstanceCreateInfo {
    s_type: StructureType,
    p_next: *const c_void,
    flags: u64,
    app_info: *const ApplicationInfo,
    enabled_layer_count: u32,
    enabled_layer: *const *const c_char,
    enabled_ext_count: u32,
    enabled_ext: *const *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Resource, Debug)]
pub struct Instance(*const c_void);

impl Default for Instance {
    fn default() -> Self {
        Instance(ptr::null())
    }
}

impl Instance {
    pub(crate) fn new() -> Insert<Instance> {
        dbg!("Starting instance initialization");
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

        let (enabled_ext_strings, enabled_ext_ptrs, enabled_ext) = c_str_array(&*ext_instance);

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
        VkResult::handle(unsafe { vkCreateInstance(&instance_info, ptr::null(), &mut instance) })
            .unwrap();
        instance.into()
    }
}
