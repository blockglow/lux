#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

layout(buffer_reference, buffer_reference_align = 4, scalar) buffer Global
{
    mat4 proj;
    mat4 view;
};

layout(push_constant, scalar) uniform Registers
{
    uint64_t global;
} registers;
layout(location = 0) out vec3 fragColor;

vec2 positions[6] = vec2[](
// First triangle
vec2(-0.5, -0.5),
vec2(0.5, -0.5),
vec2(0.5, 0.5),
// Second triangle
vec2(0.5, 0.5),
vec2(-0.5, 0.5),
vec2(-0.5, -0.5)
);

vec3 colors[6] = vec3[](
vec3(1.0, 0.0, 0.0),
vec3(0.0, 1.0, 0.0),
vec3(0.0, 0.0, 1.0),
vec3(1.0, 0.0, 0.0),
vec3(0.0, 1.0, 0.0),
vec3(0.0, 0.0, 1.0)
);

void main() {
    Global global = Global(registers.global);
    gl_Position = global.proj * global.view * vec4(positions[gl_VertexIndex], 10.0, 1.0);
    fragColor = vec3(1.0, 1.0, 0.0);
}