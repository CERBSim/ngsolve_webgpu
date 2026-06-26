// Shared declarations for Line Integral Convolution (LIC) on the 3D clipping
// plane.
//
// The clipping plane is parameterised by an orthonormal in-plane basis
// (tangent1, tangent2) and an origin (the bounding-box centre). A world point p
// lying on the plane maps to a continuous texel coordinate via
//
//     uv  = (dot(p - origin, tangent1), dot(p - origin, tangent2)) * inv_extent + 0.5
//     tex = uv * vec2(width, height)
//
// with inv_extent = 1 / (2 R), R the bounding-sphere radius, so the whole mesh
// cross-section maps into [0, 1]^2 (and thus into the texture).
//
// The same uniform is shared by every LIC stage so the parameterisation stays
// consistent between the field-evaluation compute pass, the convolution compute
// pass and the final render pass.

struct LicUniforms {
    origin: vec4f,      // xyz: plane origin in world space
    tangent1: vec4f,    // xyz: first  in-plane unit vector (texture u direction)
    tangent2: vec4f,    // xyz: second in-plane unit vector (texture v direction)
    width: u32,         // field / LIC texture width  in texels (= supersample * canvas)
    height: u32,        // field / LIC texture height in texels (= supersample * canvas)
    kernel_length: u32, // convolution kernel half-length (march steps per side, in LIC texels)
    oriented: u32,      // 0: standard LIC, 1: oriented (droplet) LIC
    thickness: u32,     // droplet size for oriented LIC, in LIC texels
    inv_extent: f32,    // world -> [0,1] scale (= 1 / (2 R))
    contrast: f32,      // LIC modulation strength in [0, 1]
    supersample: u32,   // LIC computed at this many texels per canvas pixel (SSAA)
    lic_mvp: mat4x4<f32>, // camera model-view-projection the LIC field was built
                          // with; the render pass compares it against the live
                          // u_camera MVP and skips modulation when they differ
                          // (the LIC texture is stale until Python re-dispatches).
};

@group(0) @binding(40) var<uniform> u_lic: LicUniforms;

// Continuous texel coordinate of a world point on the clipping plane.
fn licWorldToTexel(p: vec3f) -> vec2f {
    let d = p - u_lic.origin.xyz;
    let uv = vec2f(dot(d, u_lic.tangent1.xyz), dot(d, u_lic.tangent2.xyz)) * u_lic.inv_extent + 0.5;
    return uv * vec2f(f32(u_lic.width), f32(u_lic.height));
}
