#![feature(allocator_api, try_blocks, iterator_try_collect, generic_const_exprs)]
#![feature(iter_chain)]
#![feature(let_chains)]
#![no_std]

extern crate alloc as core_alloc;
pub mod input;
pub mod math;
pub mod time;
pub mod voxel;
pub mod ecs;

pub mod prelude {
	pub use core::{iter, mem, ops, ptr, slice};
	pub use core::any::*;
	pub use core::ffi::*;
	pub use core::result::*;
	pub use core_alloc::vec;

	pub use lux_macro::*;

	pub use crate::boxed::*;
	pub use crate::collections::*;
	pub use crate::ecs::*;
	pub use crate::env::{dbg, error, info, trace, warn};
	pub use crate::ffi::*;
	pub use crate::fmt::*;
	pub use crate::math::*;
	pub use crate::string::*;
	pub use crate::time::*;
	pub use crate::vec::*;
	pub use crate::ver::*;

	pub fn default<T: Default>() -> T {
        Default::default()
    }
}

mod fmt {
	pub use core_alloc::fmt;
	pub use core_alloc::format;
}

mod collections {
	pub use core_alloc::collections::*;
}

mod vec {
	pub use core_alloc::vec::*;
}

mod string {
	pub use core_alloc::string::*;
}

mod boxed {
	pub use core_alloc::boxed::*;
}

mod ffi {
	pub use core_alloc::ffi::*;
}

mod alloc {
	use core::alloc;
	use core::alloc::Layout;

	pub struct Allocator;

    #[cfg(target_os = "linux")]
    unsafe impl alloc::GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            libc::malloc(layout.size()) as *mut _
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            libc::free(ptr as *mut _);
        }
    }

    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;
}

pub mod env {
	use core::str::FromStr;

	pub use crate::prelude::*;

	pub struct Console;

    impl Console {
        pub fn log(level: Level, msg: &str) {
            let mut string = msg.to_string();
            string += "\n";
            let mut c_string = CString::from_str(&string).unwrap();
            unsafe { libc::write(1, c_string.as_ptr() as *const _, c_string.count_bytes()) };
        }
    }

    pub enum Level {
        Trace,
        Info,
        Warn,
        Error,
    }

    #[macro_export]
    macro_rules! log_dbg {
                ($($arg:tt)*) => {{
                        let var = $($arg)*;
                        let output = format!("{:?}", &var);
                        $crate::env::Console::log($crate::env::Level::Trace, &output);
                        var
                }};
        }
    pub use log_dbg as dbg;

    #[macro_export]
    macro_rules! log_trace {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Trace, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Trace, &output);
                }};
        }
    pub use log_trace as trace;

    #[macro_export]
    macro_rules! log_info {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Info, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Info, &output);
                }};
        }
    pub use log_info as info;

    #[macro_export]
    macro_rules! log_warn {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Warn, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Warn, &output);
                }};
        }
    pub use log_warn as warn;

    #[macro_export]
    macro_rules! log_error {
                () => {
                        let output = String::new();
                        $crate::env::Console::log($crate::env::Level::Error, &output);
                };
                ($($arg:tt)*) => {{
                        let output = format!($($arg)*);
                        $crate::env::Console::log($crate::env::Level::Error, &output);
                }};
        }
    pub use log_error as error;
}


pub mod gfx {
	use core::ffi::c_void;

	use vk::renderer;

	use crate::math::Matrix;
	use crate::prelude::*;
	use crate::voxel::Volume;

	#[cfg(target_os = "linux")]
    pub mod vk {

        pub mod include {
	        pub use crate::prelude::*;

	        pub use super::buffer::*;
	        pub use super::cmd;
	        pub use super::command::*;
	        pub use super::descriptor::*;
	        pub use super::device::*;
	        pub use super::ffi::*;
	        pub use super::image::*;
	        pub use super::instance::*;
	        pub use super::memory::*;
	        pub use super::physical_device::*;
	        pub use super::pipeline_layout::*;
	        pub use super::qfi::*;
	        pub use super::renderer::*;
	        pub use super::shader::*;
	        pub use super::staging::*;
	        pub use super::surface::*;
	        pub use super::swapchain::*;
	        pub use super::sync::*;
	        pub use super::util::*;
        }

        mod buffer;
        mod command;
        mod descriptor;
        mod device;
        mod ffi;
        mod image;
        mod instance;
        mod memory;
        mod physical_device;
        mod pipeline_layout;
        mod qfi;
        pub mod renderer;
        mod shader;
        mod staging;
        mod surface;
        mod swapchain;
        mod sync;
        mod util;

        pub mod cmd {}
    }

    #[cfg(target_os = "linux")]
    mod x11 {
	    use core::ffi::c_void;
	    use core::ptr;

	    use libc::c_char;

	    use crate::gfx::{Surface, SurfaceHandle};

	    pub struct Window {
            display: *const c_void,
            window: *const c_void,
        }

        impl Surface for Window {
            fn open() -> Self
            where
                Self: Sized,
            {
                let (display, window) = unsafe {
                    let display = XOpenDisplay(ptr::null());

                    let screen = XDefaultScreen(display);

                    let window = XCreateSimpleWindow(
                        display,
                        XRootWindow(display, screen),
                        10,
                        10,
                        1920,
                        1080,
                        1,
                        XBlackPixel(display, screen),
                        XWhitePixel(display, screen),
                    );
                    XMapWindow(display, window);

                    (display, window)
                };
                Self { display, window }
            }

            fn update(&self) {
                let mut buf = crate::core_alloc::vec![0u8; 65535];
                unsafe {
                    XNextEvent(self.display, buf.as_mut_ptr() as *mut _);
                }
            }

            fn size(&self) -> (u32, u32) {
                unsafe {
                    let screen = XScreenOfDisplay(self.display, 0);
                    (
                        XWidthOfScreen(screen) as u32,
                        XHeightOfScreen(screen) as u32,
                    )
                }
            }

            fn handle(&self) -> SurfaceHandle {
                SurfaceHandle::Linux {
                    display: self.display,
                    window: self.window,
                }
            }
        }

        extern "C" {
            fn XScreenOfDisplay(display: *const c_void, num: u32) -> *const c_void;
            fn XWidthOfScreen(screen: *const c_void) -> i32;
            fn XHeightOfScreen(screen: *const c_void) -> i32;
            fn XOpenDisplay(display_name: *const c_char) -> *const c_void;
            fn XBlackPixel(display: *const c_void, screen: i32) -> u64;
            fn XWhitePixel(display: *const c_void, screen: i32) -> u64;
            fn XDefaultScreen(display: *const c_void) -> i32;
            fn XRootWindow(display: *const c_void, screen: i32) -> *const c_void;
            fn XMapWindow(display: *const c_void, window: *const c_void);
            fn XCreateSimpleWindow(
                display: *const c_void,
                root: *const c_void,
                x: i32,
                y: i32,
                width: u32,
                height: u32,
                border_width: u32,
                border: u64,
                background: u64,
            ) -> *const c_void;
            fn XNextEvent(display: *const c_void, event: *mut c_void);
        }
    }

    pub fn get_platform_surface() -> Box<dyn Surface> {
        #[cfg(target_os = "linux")]
        Box::new(x11::Window::open())
    }

    pub fn get_platform_renderer() -> (Box<dyn Renderer>, Box<dyn Surface>) {
        let surface = get_platform_surface();
        #[cfg(target_os = "linux")]
        (Box::new(renderer::Renderer::create(&*surface)), surface)
    }

    pub enum SurfaceHandle {
        Linux {
            window: *const c_void,
            display: *const c_void,
        },
    }

    pub trait Surface {
        fn open() -> Self
        where
            Self: Sized;
        fn update(&self);
        fn size(&self) -> (u32, u32);
        fn handle(&self) -> SurfaceHandle;
    }

    pub trait Renderer {
        fn create(surface: &dyn Surface) -> Self
        where
            Self: Sized;
        fn render(&mut self);
        fn camera(&mut self) -> &mut Camera;
	    fn set_voxels(&mut self, id: u64, voxels: Box<dyn Volume>);
    }

    #[derive(Default, Clone, Copy)]
    pub struct Camera {
        pub proj: Matrix<4, 4>,
        pub view: Matrix<4, 4>,
    }
}

pub mod ver {
    #[repr(C)]
    pub struct Version {
        major: u8,
        minor: u8,
        patch: u8,
    }
}
