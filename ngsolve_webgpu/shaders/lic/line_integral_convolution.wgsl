// LIC stage 2: line integral convolution of the field texture.
//
// Reads the SCREEN-SPACE flow field produced by evaluate.wgsl (binding 41) and
// writes a grayscale LIC image (binding 42), both at canvas resolution. The flow
// direction at each pixel streams white noise along itself, so isocontours of the
// resulting image trace the (perspective-projected) field lines. The noise is
// hashed by framebuffer pixel, so it is crisp at any zoom but "swims" over the
// surface while the camera rotates (acceptable: static quality is the priority).
// Pixels outside the cut plane (coverage mask 0) are left blank so the final
// render can fall back to the flat colour there.

#import ngsolve/lic/common

@group(0) @binding(41) var u_lic_field: texture_2d<f32>;
// rgba8unorm (value, mask, 0, 0): filterable, so the JS engine's render-pass
// bind group (which samples this texture) accepts it; rg32float would be
// rejected as unfilterable. value and mask are both in [0, 1].
@group(0) @binding(42) var u_lic_output: texture_storage_2d<rgba8unorm, write>;

fn random(seed: u32) -> u32 {
    // Xorshift32, see https://en.wikipedia.org/wiki/Xorshift
    var value: u32 = seed;
    value ^= value << 13u;
    value ^= value >> 17u;
    value ^= value << 5u;
    return value;
}

fn randomFloat(seed: u32) -> f32 {
    return f32(random(seed)) / f32(0xFFFFFFFFu);
}

fn getSeed(j: u32, k: u32) -> u32 {
    // hash value as seed for the random number generator
    // https://www.burtleburtle.net/bob/hash/integer.html
    let n = (k << 16u) + j;
    var seed = (n ^ 61u) ^ (n >> 16u);
    seed *= seed * 9u;
    seed = seed ^ (seed >> 4u);
    seed = seed * 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

fn noise(x: u32, y: u32) -> f32 {
    return randomFloat(getSeed(x, y));
}

fn dropletNoise(x: u32, y: u32) -> f32 {
    // large droplets for oriented line integral convolution
    let n = max(u_lic.thickness, 1u);
    let rand: f32 = noise(x / n, y / n);
    if (rand < 0.85) {
        return 0.0;
    }
    let dx = f32(x % n) - f32(n) * 0.5;
    let dy = f32(y % n) - f32(n) * 0.5;
    // Smooth (C1) radial falloff instead of a hard quadratic clip, so a droplet
    // reads as a soft blob. Combined with the LIC being supersampled and box-
    // downsampled in the render pass, this antialiases thin oriented streaks
    // (the cause of the jaggies at small thickness). r is the distance from the
    // cell centre normalised so r = 1 at the cell edge (radius n/2).
    let r = clamp(sqrt(dx * dx + dy * dy) / (f32(n) * 0.5), 0.0, 1.0);
    return 1.0 - smoothstep(0.0, 1.0, r);
}

fn flowAt(p: vec2f) -> vec4f {
    return textureLoad(u_lic_field, vec2<i32>(p), 0);
}

fn lineIntegralConvolution(x: u32, y: u32) -> f32 {
    let w = u_lic.width;
    let h = u_lic.height;
    let kernel_length = u_lic.kernel_length;
    let oriented = u_lic.oriented;
    var sum: f32 = 0.0;
    var weight: f32 = 0.0;

    for (var dir: i32 = -1; dir <= 1; dir += 2) {
        var p = vec2f(f32(x) + 0.5, f32(y) + 0.5);

        for (var k: u32 = 0u; k < kernel_length; k++) {
            let sample = flowAt(p);
            // Stop at the plane boundary or at a no-value (NaN) point so the
            // streamline does not bleed past the data. NaN != NaN.
            if (sample.w < 0.5 || sample.x != sample.x || sample.y != sample.y) {
                break;
            }
            var v = sample.xy;
            let len = length(v);
            if (len < 1e-8) {
                break;  // stagnation point
            }
            v = v / len;

            p += f32(dir) * v;

            if (p.x < 0.0 || p.x >= f32(w) || p.y < 0.0 || p.y >= f32(h)) {
                break;
            }

            let ix = u32(p.x);
            let iy = u32(p.y);

            if (oriented == 0u) {
                let kernel_weight = 1.0 - f32(k) / f32(kernel_length);
                sum += kernel_weight * noise(ix, iy);
                weight += kernel_weight;
            } else {
                // The asymmetric kernel weights one march direction more
                // heavily, which is what gives oriented LIC its visible flow
                // direction. (1 - dir*...) points the bright "tail" the correct
                // way along the flow; (1 + dir*...) reverses it.
                let t = 0.5 * (1.0 - f32(dir) * f32(k) / f32(kernel_length));
                let kernel_weight = 0.1 + 0.9 * t * t * t * t;
                sum += kernel_weight * dropletNoise(ix, iy);
                weight += kernel_weight;
            }
        }
    }

    if (weight <= 0.0) {
        return 0.5;
    }
    var result = sum / weight;
    if (oriented != 0u) {
        // Oriented LIC convolves sparse droplet noise (zero in ~85% of cells),
        // so its weighted average is far darker than the standard mode's uniform
        // noise (~0.5 mean). Lift it with a gain so the streaks read at a similar
        // brightness; clamp keeps the value in the texture's [0, 1] range.
        result = clamp(result * 6.0, 0.0, 1.0);
    }
    return result;
}

@compute @workgroup_size(16, 16, 1)
fn compute_lic(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u_lic.width || gid.y >= u_lic.height) {
        return;
    }
    let mask = textureLoad(u_lic_field, vec2<i32>(gid.xy), 0).w;
    if (mask < 0.5) {
        textureStore(u_lic_output, vec2<i32>(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
        return;
    }
    let value = lineIntegralConvolution(gid.x, gid.y);
    textureStore(u_lic_output, vec2<i32>(gid.xy), vec4f(value, 1.0, 0.0, 0.0));
}
