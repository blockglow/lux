#![feature(generic_const_exprs)]
#![no_std]
#![no_main]

use core::panic::PanicInfo;

use lux::gfx::{Camera, get_platform_renderer};
use lux::input::controller::{get_platform_controller, Xbox};
use lux::math::Projection;
use lux::prelude::*;

#[no_mangle]
pub fn main() {
    let (mut renderer, surface) = get_platform_renderer();
    let mut controller = get_platform_controller(0);
    let mut translation = Vector([0.0,0.0,6.0]);
    let mut facing = Vector([0.0, 0.0, 0.0]);
    let mut rotation = Quaternion::default();
    loop {

        let movement = Vector([
            controller.get_axis(Xbox::L_STICK_X),
            controller.get_button_axis(Xbox::R_STICK, Xbox::L_STICK),
            -controller.get_axis(Xbox::L_STICK_Y),
        ]).deadzone();
        translation += rotation * movement * 0.01;

        facing = Vector([
            controller.get_button_axis(Xbox::R_BUMPER, Xbox::L_BUMPER) * 0.005,
            -controller.get_axis(Xbox::R_STICK_Y) * 0.005,
            controller.get_axis(Xbox::R_STICK_X) * 0.005,
        ]);

        let pitch = Quaternion::from_angle_axis(facing.0[1], rotation * Vector::X).unwrap();
        let yaw = Quaternion::from_angle_axis(facing.0[2], rotation * Vector::Y).unwrap();
        let roll = Quaternion::from_angle_axis(facing.0[0], rotation * Vector::Z).unwrap();

        rotation = yaw * pitch * roll * rotation;

        let trans = rotation.to_matrix() * Matrix::from_translation(translation);

        controller.poll();
        renderer.render();
        let (w, h) = surface.size();
        let a = w as f32 / h as f32;
        *renderer.camera() = Camera {
            proj: (Projection { near: 0.1, far: 1000.0, a, ..Projection::default()}).into(),
            view: trans.inverse().unwrap(),
        };

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
