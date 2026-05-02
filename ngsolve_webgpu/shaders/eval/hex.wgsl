#import ngsolve/eval/common

const N_DOFS_HEX_VEC3: u32 = (@MAX_EVAL_ORDER_VEC3@+1) * (@MAX_EVAL_ORDER_VEC3@+1) * (@MAX_EVAL_ORDER_VEC3@+1);

// Tensor-product de Casteljau evaluation of vec3-valued Bernstein on a hex.
// Storage order: ix + n1*iy + n1*n1*iz, each DOF is 3 consecutive f32.
// Reference coords lam in [0,1]^3.
fn evalHexVec3At(coeff_start: u32, order: i32, lam: vec3f) -> vec3f {
    let n1 = order + 1;
    let ndof = u32(n1 * n1 * n1);

    var v: array<vec3f, N_DOFS_HEX_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_start + i * 3u;
        v[i] = vec3f(mesh.data[base], mesh.data[base + 1u], mesh.data[base + 2u]);
    }

    // Phase 1: reduce z
    for (var level = 1; level <= order; level++) {
        for (var iz = 0; iz <= order - level; iz++) {
            for (var iy = 0; iy <= order; iy++) {
                for (var ix = 0; ix <= order; ix++) {
                    let idx0 = ix + n1 * iy + n1 * n1 * iz;
                    let idx1 = ix + n1 * iy + n1 * n1 * (iz + 1);
                    v[idx0] = (1.0 - lam.z) * v[idx0] + lam.z * v[idx1];
                }
            }
        }
    }

    // Phase 2: reduce y
    for (var level = 1; level <= order; level++) {
        for (var iy = 0; iy <= order - level; iy++) {
            for (var ix = 0; ix <= order; ix++) {
                let idx0 = ix + n1 * iy;
                let idx1 = ix + n1 * (iy + 1);
                v[idx0] = (1.0 - lam.y) * v[idx0] + lam.y * v[idx1];
            }
        }
    }

    // Phase 3: reduce x
    for (var level = 1; level <= order; level++) {
        for (var ix = 0; ix <= order - level; ix++) {
            v[ix] = (1.0 - lam.x) * v[ix] + lam.x * v[ix + 1];
        }
    }

    return v[0];
}

// Gradient: returns mat4x3f(pos, dF/dx, dF/dy, dF/dz)
fn evalHexVec3GradAt(coeff_start: u32, order: i32, lam: vec3f) -> mat4x3f {
    let n1 = order + 1;
    let ndof = u32(n1 * n1 * n1);

    var v: array<vec3f, N_DOFS_HEX_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_start + i * 3u;
        v[i] = vec3f(mesh.data[base], mesh.data[base + 1u], mesh.data[base + 2u]);
    }

    // --- dF/dz: z-differences, then full 3-phase de Casteljau on (n-1,n,n) ---
    var vdz: array<vec3f, N_DOFS_HEX_VEC3>;
    for (var iz = 0; iz < order; iz++) {
        for (var iy = 0; iy <= order; iy++) {
            for (var ix = 0; ix <= order; ix++) {
                let idx0 = ix + n1 * iy + n1 * n1 * iz;
                let idx1 = ix + n1 * iy + n1 * n1 * (iz + 1);
                vdz[idx0] = v[idx1] - v[idx0];
            }
        }
    }
    // reduce z (order-1 levels)
    for (var level = 1; level < order; level++) {
        for (var iz = 0; iz < order - level; iz++) {
            for (var iy = 0; iy <= order; iy++) {
                for (var ix = 0; ix <= order; ix++) {
                    let idx0 = ix + n1 * iy + n1 * n1 * iz;
                    let idx1 = ix + n1 * iy + n1 * n1 * (iz + 1);
                    vdz[idx0] = (1.0 - lam.z) * vdz[idx0] + lam.z * vdz[idx1];
                }
            }
        }
    }
    // reduce y
    for (var level = 1; level <= order; level++) {
        for (var iy = 0; iy <= order - level; iy++) {
            for (var ix = 0; ix <= order; ix++) {
                let idx0 = ix + n1 * iy;
                let idx1 = ix + n1 * (iy + 1);
                vdz[idx0] = (1.0 - lam.y) * vdz[idx0] + lam.y * vdz[idx1];
            }
        }
    }
    // reduce x
    for (var level = 1; level <= order; level++) {
        for (var ix = 0; ix <= order - level; ix++) {
            vdz[ix] = (1.0 - lam.x) * vdz[ix] + lam.x * vdz[ix + 1];
        }
    }
    let dz = vdz[0] * f32(order);

    // --- Reduce v along z (Phase 1) ---
    for (var level = 1; level <= order; level++) {
        for (var iz = 0; iz <= order - level; iz++) {
            for (var iy = 0; iy <= order; iy++) {
                for (var ix = 0; ix <= order; ix++) {
                    let idx0 = ix + n1 * iy + n1 * n1 * iz;
                    let idx1 = ix + n1 * iy + n1 * n1 * (iz + 1);
                    v[idx0] = (1.0 - lam.z) * v[idx0] + lam.z * v[idx1];
                }
            }
        }
    }
    // Now v[ix + n1*iy] for ix,iy in 0..order holds the z-reduced (n+1)^2 values

    // --- dF/dy: y-differences from z-reduced, then 2-phase de Casteljau ---
    var vdy: array<vec3f, N_DOFS_HEX_VEC3>;
    for (var iy = 0; iy < order; iy++) {
        for (var ix = 0; ix <= order; ix++) {
            let idx0 = ix + n1 * iy;
            let idx1 = ix + n1 * (iy + 1);
            vdy[idx0] = v[idx1] - v[idx0];
        }
    }
    // reduce y (order-1 levels)
    for (var level = 1; level < order; level++) {
        for (var iy = 0; iy < order - level; iy++) {
            for (var ix = 0; ix <= order; ix++) {
                let idx0 = ix + n1 * iy;
                let idx1 = ix + n1 * (iy + 1);
                vdy[idx0] = (1.0 - lam.y) * vdy[idx0] + lam.y * vdy[idx1];
            }
        }
    }
    // reduce x
    for (var level = 1; level <= order; level++) {
        for (var ix = 0; ix <= order - level; ix++) {
            vdy[ix] = (1.0 - lam.x) * vdy[ix] + lam.x * vdy[ix + 1];
        }
    }
    let dy = vdy[0] * f32(order);

    // --- Reduce v along y (Phase 2) ---
    for (var level = 1; level <= order; level++) {
        for (var iy = 0; iy <= order - level; iy++) {
            for (var ix = 0; ix <= order; ix++) {
                let idx0 = ix + n1 * iy;
                let idx1 = ix + n1 * (iy + 1);
                v[idx0] = (1.0 - lam.y) * v[idx0] + lam.y * v[idx1];
            }
        }
    }
    // Now v[ix] for ix in 0..order holds the y-reduced values

    // --- dF/dx: x-differences, then 1-phase de Casteljau ---
    var vdx: array<vec3f, N_DOFS_HEX_VEC3>;
    for (var ix = 0; ix < order; ix++) {
        vdx[ix] = v[ix + 1] - v[ix];
    }
    // reduce x (order-1 levels)
    for (var level = 1; level < order; level++) {
        for (var ix = 0; ix < order - level; ix++) {
            vdx[ix] = (1.0 - lam.x) * vdx[ix] + lam.x * vdx[ix + 1];
        }
    }
    let dx = vdx[0] * f32(order);

    // --- Position: reduce v along x (Phase 3) ---
    for (var level = 1; level <= order; level++) {
        for (var ix = 0; ix <= order - level; ix++) {
            v[ix] = (1.0 - lam.x) * v[ix] + lam.x * v[ix + 1];
        }
    }
    let pos = v[0];

    return mat4x3f(pos, dx, dy, dz);
}
