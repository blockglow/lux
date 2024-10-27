#![feature(generic_const_exprs)]

use lux::gfx::vk::Vulkan;
use lux::gfx::PlatformSurface;
pub use lux::prelude::*;

pub fn main() {
    App::default().add(PlatformSurface).add(Vulkan {}).run();
}
