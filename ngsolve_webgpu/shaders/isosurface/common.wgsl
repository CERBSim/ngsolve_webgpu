#import ngsolve/clipping/common

struct IsosurfaceUniforms {
    subdivision: u32,
    value: f32,
    padding: f32,
    padding1: f32,
};

@group(0) @binding(27) var<uniform> u_iso: IsosurfaceUniforms;
