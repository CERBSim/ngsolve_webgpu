@group(0) @binding(13) var<storage> u_function_values_3d : array<f32>;
@group(0) @binding(10) var<storage> u_function_values_2d : array<f32>;
// @group(0) @binding(14) var<storage> u_curvature_values_2d : array<f32>;
@group(0) @binding(15) var<uniform> u_subdivision : u32;
@group(0) @binding(16) var<storage> u_deformation_values_2d : array<f32>;
@group(0) @binding(17) var<uniform> u_deformation_scale : f32;
@group(0) @binding(18) var<storage> u_deformation_values_3d : array<f32>;
@group(0) @binding(55) var<uniform> u_function_component: i32;

struct ComplexSettings {
    mode: u32,    // 0=PhaseRotate, 1=Abs, 2=Arg
    phase: f32,   // phase angle for mode 0: Re(z * e^{i*phase})
    padding: vec2f,
};
@group(0) @binding(56) var<uniform> u_complex: ComplexSettings;

// storing number of components, order, and is_complex flag in first three entries
const VALUES_OFFSET: u32 = 3;

