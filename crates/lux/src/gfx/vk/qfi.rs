use crate::prelude::*;

use super::include::*;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct QueueFamilyIndex(u32);

impl QueueFamilyIndex {
    pub fn queue(self, device: Device) -> Queue {
        let mut queue = Queue::default();
        unsafe { vkGetDeviceQueue(device, self, 0, &mut queue) };
        queue
    }
    pub fn graphics(physical_device: PhysicalDevice, surface: Surface) -> Result<Self, Error> {
        let mut count = 8;
        let mut queue_family_properties =
            vec![unsafe { mem::zeroed::<QueueFamilyProperties>() }; 8];
        unsafe {
            vkGetPhysicalDeviceQueueFamilyProperties(
                physical_device,
                &mut count,
                queue_family_properties.as_mut_ptr(),
            )
        };

        let queue_family_index = queue_family_properties
            .iter()
            .enumerate()
            .find(|(_index, properties)| {
                let graphics = properties.queue_flags & QueueFlagBits::Graphics as u32 != 0;
                let mut present = false as _;
                unsafe {
                    vkGetPhysicalDeviceSurfaceSupportKHR(
                        physical_device,
                        *_index as u32,
                        surface,
                        &mut present,
                    )
                };
                let present = present == true as _;
                graphics && present
            })
            .map(|(index, _properties)| index as u32)
            .unwrap();

        Ok(Self(queue_family_index))
    }
}
