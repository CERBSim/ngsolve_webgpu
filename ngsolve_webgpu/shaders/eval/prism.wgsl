#import ngsolve/eval/common

const N_DOFS_PRISM_VEC3: u32 = (@MAX_EVAL_ORDER_VEC3@+1) * (@MAX_EVAL_ORDER_VEC3@+2) / 2 * (@MAX_EVAL_ORDER_VEC3@+1);

// 2D triangle de Casteljau for vec3 values.
// Barycentrics: b = (lam.x, lam.y, 1 - lam.x - lam.y)
// Uses the same in-place overwrite trick as the tet version.
fn _evalTrigVec3DeCasteljau(v: ptr<function, array<vec3f, N_DOFS_PRISM_VEC3>>, order: i32, lam: vec2f) -> vec3f {
    let b = vec3f(lam, 1.0 - lam.x - lam.y);

    for (var n = order; n > 0; n--) {
        var idx = 0;
        for (var iy = 0; iy < n; iy++) {
            let stride_y = order + 1 - iy;
            for (var ix = 0; ix < n - iy; ix++) {
                let p = idx + ix;
                (*v)[p] = b.x * (*v)[p] + b.y * (*v)[p + 1] + b.z * (*v)[p + stride_y];
            }
            idx += order + 1 - iy;
        }
        // skip unprocessed rows
        for (var iy = n; iy <= order; iy++) {
            idx += order + 1 - iy;
        }
    }

    return (*v)[0];
}

// Prism Bernstein evaluation: triangle(x,y) x 1D(z).
// Storage: z-layer major, triangle-DOF minor.
// lam = (x, y, z) where (x,y) is triangle coord (x+y<=1), z in [0,1].
fn evalPrismVec3At(coeff_start: u32, order: i32, lam: vec3f) -> vec3f {
    let ndof_trig = (order + 1) * (order + 2) / 2;
    let ndof = u32(ndof_trig * (order + 1));

    var v: array<vec3f, N_DOFS_PRISM_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_start + i * 3u;
        v[i] = vec3f(mesh.data[base], mesh.data[base + 1u], mesh.data[base + 2u]);
    }

    // Reduce z: 1D de Casteljau over z-layers for each triangle DOF
    for (var level = 1; level <= order; level++) {
        for (var l = 0; l <= order - level; l++) {
            for (var t = 0; t < ndof_trig; t++) {
                let i0 = l * ndof_trig + t;
                let i1 = (l + 1) * ndof_trig + t;
                v[i0] = (1.0 - lam.z) * v[i0] + lam.z * v[i1];
            }
        }
    }

    // Now v[0..ndof_trig-1] is the z-reduced triangle polynomial
    // Apply 2D triangle de Casteljau
    return _evalTrigVec3DeCasteljau(&v, order, vec2f(lam.x, lam.y));
}

// Gradient: returns mat4x3f(pos, dF/dx, dF/dy, dF/dz)
fn evalPrismVec3GradAt(coeff_start: u32, order: i32, lam: vec3f) -> mat4x3f {
    let ndof_trig = (order + 1) * (order + 2) / 2;
    let ndof = u32(ndof_trig * (order + 1));

    var v: array<vec3f, N_DOFS_PRISM_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_start + i * 3u;
        v[i] = vec3f(mesh.data[base], mesh.data[base + 1u], mesh.data[base + 2u]);
    }

    // --- dF/dz: z-differences for each trig DOF, then reduce ---
    var vdz: array<vec3f, N_DOFS_PRISM_VEC3>;
    for (var l = 0; l < order; l++) {
        for (var t = 0; t < ndof_trig; t++) {
            let i0 = l * ndof_trig + t;
            let i1 = (l + 1) * ndof_trig + t;
            vdz[i0] = v[i1] - v[i0];
        }
    }
    // reduce z (order-1 levels) on the difference array
    for (var level = 1; level < order; level++) {
        for (var l = 0; l < order - level; l++) {
            for (var t = 0; t < ndof_trig; t++) {
                let i0 = l * ndof_trig + t;
                let i1 = (l + 1) * ndof_trig + t;
                vdz[i0] = (1.0 - lam.z) * vdz[i0] + lam.z * vdz[i1];
            }
        }
    }
    // triangle de Casteljau on the z-reduced differences
    let dz = _evalTrigVec3DeCasteljau(&vdz, order, vec2f(lam.x, lam.y)) * f32(order);

    // --- Reduce v along z ---
    for (var level = 1; level <= order; level++) {
        for (var l = 0; l <= order - level; l++) {
            for (var t = 0; t < ndof_trig; t++) {
                let i0 = l * ndof_trig + t;
                let i1 = (l + 1) * ndof_trig + t;
                v[i0] = (1.0 - lam.z) * v[i0] + lam.z * v[i1];
            }
        }
    }
    // Now v[0..ndof_trig-1] holds the z-reduced triangle coefficients

    // --- dF/dx, dF/dy from the triangle coefficients ---
    // Differences: dx[p] = v[p] - v[p + stride_y], dy[p] = v[p+1] - v[p + stride_y]
    var vdx: array<vec3f, N_DOFS_PRISM_VEC3>;
    var vdy: array<vec3f, N_DOFS_PRISM_VEC3>;
    var src_idx = 0;
    var dst_idx = 0;
    for (var iy = 0; iy < order; iy++) {
        let stride_y = order + 1 - iy;
        for (var ix = 0; ix < order - iy; ix++) {
            let p = src_idx + ix;
            vdx[dst_idx] = v[p] - v[p + stride_y];
            vdy[dst_idx] = v[p + 1] - v[p + stride_y];
            dst_idx++;
        }
        src_idx += order + 1 - iy;
    }
    let dx = _evalTrigVec3DeCasteljau(&vdx, order - 1, vec2f(lam.x, lam.y)) * f32(order);
    let dy = _evalTrigVec3DeCasteljau(&vdy, order - 1, vec2f(lam.x, lam.y)) * f32(order);

    // --- Position: triangle de Casteljau on the z-reduced v ---
    let pos = _evalTrigVec3DeCasteljau(&v, order, vec2f(lam.x, lam.y));

    return mat4x3f(pos, dx, dy, dz);
}
