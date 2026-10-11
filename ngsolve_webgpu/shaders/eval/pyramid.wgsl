#import ngsolve/eval/hex

// The pyramid map is a hex Bernstein polynomial in the collapsed coordinates
// (x/(1-z), y/(1-z), z), see get_pyramid_intrule.
fn _pyramidCollapsed(lam: vec3f) -> vec3f {
    let noz = max(1.0 - lam.z, 1e-6);
    return vec3f(clamp(lam.x / noz, 0.0, 1.0), clamp(lam.y / noz, 0.0, 1.0), lam.z);
}

fn evalPyramidVec3At(coeff_start: u32, order: i32, lam: vec3f) -> vec3f {
    return evalHexVec3At(coeff_start, order, _pyramidCollapsed(lam));
}

// Gradient: returns mat4x3f(pos, dF/dx, dF/dy, dF/dz)
fn evalPyramidVec3GradAt(coeff_start: u32, order: i32, lam: vec3f) -> mat4x3f {
    let noz = max(1.0 - lam.z, 1e-6);
    let c = _pyramidCollapsed(lam);
    let g = evalHexVec3GradAt(coeff_start, order, c);
    return mat4x3f(g[0], g[1] / noz, g[2] / noz, g[3] + (c.x * g[1] + c.y * g[2]) / noz);
}
