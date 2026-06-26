// Render-side LIC modulation, shared by clipping/render.wgsl and mesh/render.wgsl.
//
// Declares the sampled LIC output texture (binding 42, read side) and the
// box-downsample + coverage-weighted modulation used in both #ifdef LIC
// branches. Both render shaders #import this instead of declaring binding 42
// inline: the engine concatenates every render object's shader code into one
// module, so two inline declarations (one per LIC render object) collided as a
// redeclaration of u_lic_output. #import is deduplicated, so a single shared
// declaration is emitted no matter how many LIC objects are in the scene.
//
// NOTE: binding 42 is the *read* (texture_2d<f32>) view here. The LIC compute
// pass (lic/line_integral_convolution.wgsl) declares the same binding as a
// write storage texture in its own module, so this declaration must NOT move
// into lic/common.wgsl (which the compute pass also imports).
#import camera
#import ngsolve/lic/common
@group(0) @binding(42) var u_lic_output: texture_2d<f32>;

// True when the LIC texture is stale: the live camera (u_camera) no longer
// matches the camera the LIC field was rasterised with (u_lic.lic_mvp). In the
// live JS engine a camera drag re-renders GPU-side without re-running the
// Python-dispatched LIC compute, so the streamlines would be misaligned with
// the moved view until the (debounced) re-dispatch. While stale, the render
// pass shows the plain colormap instead of a wrong modulation. The epsilon
// absorbs float / Python-vs-engine MVP differences; the LIC is only ever
// "fresh" right after a dispatch, where the two MVPs are identical.
fn licIsStale() -> bool {
    let m = u_camera.model_view_projection;
    let s = u_lic.lic_mvp;
    let d = abs(m[0] - s[0]) + abs(m[1] - s[1]) + abs(m[2] - s[2]) + abs(m[3] - s[3]);
    return (d.x + d.y + d.z + d.w) > 1e-3;
}

// Modulate a colormapped fragment colour with the precomputed screen-space LIC
// grayscale. The LIC texture is supersampled at u_lic.supersample texels per
// canvas pixel, so this box-averages the supersample x supersample block under
// the fragment's framebuffer pixel (the antialiasing / fractional-width
// resolve). The value is weighted by the coverage mask (channel .y, 0 at gaps /
// no-value), so partial-coverage edges blend and gaps fall back to the flat
// colour.
fn licModulate(color: vec4f, fragPosition: vec2f) -> vec4f {
    // Stale LIC (camera moved since the last dispatch): don't sample the
    // misaligned streamlines, but approximate the LIC's *average* darkening so
    // the plane doesn't brighten while moving and then pop dark when the fresh
    // LIC arrives. The LIC grayscale averages ~0.5, so the mean shade is
    // mix(1, 0.5, contrast) = 0.5 + (0.5 - 0.5*contrast).
    if (licIsStale()) {
        let shade = 0.5 + (0.5 - 0.5 * u_lic.contrast);
        return vec4f(color.rgb * shade, color.a);
    }
    let ss = max(i32(u_lic.supersample), 1);
    let base = vec2<i32>(floor(fragPosition)) * ss;
    let maxxy = vec2<i32>(i32(u_lic.width) - 1, i32(u_lic.height) - 1);
    var sum_v = 0.0;
    var sum_m = 0.0;
    for (var dy = 0; dy < ss; dy++) {
      for (var dx = 0; dx < ss; dx++) {
        let s = textureLoad(u_lic_output, clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), maxxy), 0);
        sum_v += s.x * s.y;
        sum_m += s.y;
      }
    }
    let cover = sum_m / f32(ss * ss);
    let licv = clamp(1.5 * sum_v / max(sum_m, 1e-6), 0.0, 1.0);
    let shade = mix(1.0, licv, clamp(u_lic.contrast * cover, 0.0, 1.0));
    return vec4f(color.rgb * shade, color.a);
}
