use std::process::Command;

pub fn main() {
    println!("cargo::rustc-link-lib=X11");
    println!("cargo::rustc-link-lib=vulkan");
    println!("cargo::rerun-if-changed=src/shader/*");
    Command::new("glslc")
        .args([
            "-fshader-stage=vert",
            "/home/solmidnight/work/lux/crates/lux/src/shader/vertex.glsl",
            "-o",
            "/home/solmidnight/work/lux/assets/shader/vertex.spirv",
        ])
        .spawn()
        .unwrap();
    Command::new("glslc")
        .args([
            "-fshader-stage=frag",
            "/home/solmidnight/work/lux/crates/lux/src/shader/fragment.glsl",
            "-o",
            "/home/solmidnight/work/lux/assets/shader/fragment.spirv",
        ])
        .spawn()
        .unwrap();
}
