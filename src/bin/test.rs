#![no_std]
#![no_main]

use core::panic::PanicInfo;

use libc::sleep;

use lux::gfx::make_vk_surface;
use lux::gfx::vk::Context;
use lux::prelude::*;

#[no_mangle]
pub fn main() {
        let surface = make_vk_surface();
        let ctx = Context::acquire(&*surface);
        loop {
                info!("deez");
                surface.update();
                unsafe { sleep(1); }
        }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
        info!("{info}");
        loop {}
}

#[no_mangle]
extern "C" fn _Unwind_Resume() -> ! {
        unreachable!("Unwinding not supported");
}

#[no_mangle]
extern "C" fn rust_eh_personality() -> ! {
        unreachable!("personality not supported");
}
