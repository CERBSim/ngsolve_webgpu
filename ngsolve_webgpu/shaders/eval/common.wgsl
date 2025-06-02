@group(0) @binding(13) var<storage> u_function_values_3d : array<f32>;
@group(0) @binding(10) var<storage> u_function_values_2d : array<f32>;
@group(0) @binding(14) var<storage> u_curvature_values_2d : array<f32>;
@group(0) @binding(15) var<uniform> u_subdivision : u32;
@group(0) @binding(16) var<storage> u_deformation_values_2d : array<f32>;
@group(0) @binding(17) var<uniform> u_deformation_scale : f32;
@group(0) @binding(55) var<storage> u_function_component: i32;



// storing number of components and order of basis functions in first two entries
const VALUES_OFFSET: u32 = 2; 

