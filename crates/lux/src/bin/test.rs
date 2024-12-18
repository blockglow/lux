#![feature(generic_const_exprs)]
#![no_std]
#![no_main]

use core::f32::consts::PI;
use core::panic::PanicInfo;

use lux::ecs::World;
use lux::gfx::get_platform_renderer;
use lux::input::controller::{get_platform_controller, Xbox};
use lux::input::Input;
use lux::math::Projection;
use lux::prelude::*;
use lux::time::{Instant, Span};

#[derive(Component)]
struct Player;

#[derive(Component, Default)]
struct Transform(lux::math::Transform);

#[derive(Component, Default)]
struct Camera {
    projection: Projection,
}

fn player_input(input: Input, query: Query<(&mut Transform,), With<Player>>) {
    let transform = query.single();
    let controller = input.get_controller();

    let movement = Vector([
        controller.get_axis(Xbox::L_STICK_X),
        controller.get_button_axis(Xbox::L_STICK, Xbox::R_STICK),
        -controller.get_axis(Xbox::L_STICK_Y),
    ])
    .deadzone();

    let facing = Vector([
        controller.get_button_axis(Xbox::R_BUMPER, Xbox::L_BUMPER),
        -controller.get_axis(Xbox::R_STICK_Y),
        controller.get_axis(Xbox::R_STICK_X),
    ]) * PI
        * (1.0 / 3.0)
        * delta;

    let pitch = Quaternion::from_angle_axis(facing.0[1], rotation * Vector::X).unwrap();
    let yaw = Quaternion::from_angle_axis(facing.0[2], rotation * Vector::Y).unwrap();
    let roll = Quaternion::from_angle_axis(facing.0[0], rotation * Vector::Z).unwrap();

    transform.rotation = yaw * pitch * roll * rotation;
    transform.translation += rotation * movement * delta * 5.0;
}

#[no_mangle]
pub fn main() {
    let mut world = World::default();
    world.push((Player, Transform::default(), Camera::default()));

    let (mut renderer, surface) = get_platform_renderer();
    let mut controller = get_platform_controller(0);
    let start = Instant::now();
    let mut last = start;
    loop {
        let now = Instant::now();
        let delta = (now - last).as_secs();
        last = now;

        let trans = rotation.to_matrix() * Matrix::from_translation(translation);

        controller.poll();
        renderer.render();
        let (w, h) = surface.size();
        let a = w as f32 / h as f32;
        *renderer.camera() = Camera {
            proj: (Projection {
                near: 0.1,
                far: 1000.0,
                a,
                ..Projection::default()
            })
            .into(),
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
