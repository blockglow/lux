glslc -fshader-stage=vert ./crates/lux/src/shader/vertex.glsl  -o ./assets/shader/vertex.spirv
glslc -fshader-stage=frag ./crates/lux/src/shader/fragment.glsl -o ./assets/shader/fragment.spirv
