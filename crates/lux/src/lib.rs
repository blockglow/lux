#![feature(allocator_api, try_blocks, iterator_try_collect, generic_const_exprs)]
#![feature(iter_chain, try_trait_v2)]
#![feature(let_chains)]

extern crate alloc as core_alloc;
extern crate core;

pub mod ecs;
pub mod input;
pub mod math;
pub mod time;
pub mod voxel;

pub mod prelude {
    pub use core::any::*;
    pub use core::ffi::*;
    pub use core::result::*;
    pub use core::{iter, mem, ops, ptr, slice};
    pub use core_alloc::vec;

    pub use lux_macro::*;

    pub use crate::boxed::*;
    pub use crate::collections::*;
    pub use crate::ecs::Condition::*;
    pub use crate::ecs::*;
    pub use crate::env::{error, info, trace, warn};
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
    use core::ops::{Deref, DerefMut};

    use vk::renderer2;

    use crate::math::Matrix;
    use crate::prelude::*;
    use crate::voxel::Volume;

    #[cfg(target_os = "linux")]
    pub mod vk;

    #[cfg(target_os = "linux")]
    pub mod x11;

    pub struct PlatformSurface;
    impl PlatformSurface {
        pub fn get() -> ActiveSurface {
            #[cfg(target_os = "linux")]
            ActiveSurface(Box::new(x11::Window::open()))
        }
    }

    impl Plugin for PlatformSurface {
        fn add(self, app: App) -> App {
            app.insert(Self::get())
        }
    }

    pub enum SurfaceHandle {
        Linux {
            window: *const c_void,
            display: *const c_void,
        },
    }

    #[derive(Resource)]
    pub struct ActiveSurface(Box<dyn Surface>);

    impl Deref for ActiveSurface {
        type Target = dyn Surface;

        fn deref(&self) -> &Self::Target {
            &*self.0
        }
    }

    impl DerefMut for ActiveSurface {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut *self.0
        }
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

    #[derive(Resource, Default, Clone, Copy)]
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
