#import ngsolve/eval/common

const N_DOFS_TRIG: u32 = (@MAX_EVAL_ORDER@+1) * (@MAX_EVAL_ORDER@ + 2) / 2;
const N_DOFS_TRIG_VEC3: u32 = (@MAX_EVAL_ORDER_VEC3@+1) * (@MAX_EVAL_ORDER_VEC3@ + 2) / 2;

fn evalTrig(data: ptr<storage, array<f32>, read>, id: u32, icomp: i32, lam: vec2<f32>) -> f32 {
    var order: i32 = i32((*data)[1]);
    let ncomp: u32 = u32((*data)[0]);
    let is_complex: u32 = u32((*data)[2]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    let stride: u32 = ncomp * (1u + is_complex);
    let offset: u32 = ndof * id * stride + VALUES_OFFSET;

    if is_complex == 0u {
        // Real path
        var v: array<f32, N_DOFS_TRIG>;
        if icomp == -1 {
            for (var i: u32 = 0u; i < ndof; i++) {
                v[i] = 0.0;
                for (var j: u32 = 0u; j < ncomp; j++) {
                    v[i] += (*data)[offset + i * stride + j] * (*data)[offset + i * stride + j];
                }
                v[i] = sqrt(v[i]);
            }
        } else {
            for (var i: u32 = 0u; i < ndof; i++) {
                v[i] = (*data)[offset + u32(icomp) + i * stride];
            }
        }

        let dy = order + 1;
        let b = vec3f(lam.x, lam.y, 1.0 - lam.x - lam.y);
        for (var n = order; n > 0; n--) {
            var i0 = 0;
            for (var iy = 0; iy < n; iy++) {
                for (var ix = 0; ix < n - iy; ix++) {
                    v[i0 + ix] = dot(b, vec3f(v[i0 + ix], v[i0 + ix + 1], v[i0 + ix + dy - iy]));
                }
                i0 += dy - iy;
            }
        }
        return v[0];
    }

    // Complex path: evaluate Re and Im independently via de Casteljau
    var v_re: array<f32, N_DOFS_TRIG>;
    var v_im: array<f32, N_DOFS_TRIG>;

    if icomp == -1 {
        // norm: sqrt(sum_k(re_k^2 + im_k^2))
        for (var i: u32 = 0u; i < ndof; i++) {
            v_re[i] = 0.0;
            v_im[i] = 0.0;
            for (var j: u32 = 0u; j < ncomp; j++) {
                let re = (*data)[offset + i * stride + j * 2u];
                let im = (*data)[offset + i * stride + j * 2u + 1u];
                v_re[i] += re * re + im * im;
            }
            v_re[i] = sqrt(v_re[i]);
            // For norm mode, we compute sqrt(sum |z_k|^2) at each DOF,
            // then run de Casteljau on that (approximation).
            // v_im stays 0 - we only use v_re.
        }
    } else {
        let comp_offset = u32(icomp) * 2u;
        for (var i: u32 = 0u; i < ndof; i++) {
            v_re[i] = (*data)[offset + i * stride + comp_offset];
            v_im[i] = (*data)[offset + i * stride + comp_offset + 1u];
        }
    }

    let dy = order + 1;
    let b = vec3f(lam.x, lam.y, 1.0 - lam.x - lam.y);

    // De Casteljau on real part
    for (var n = order; n > 0; n--) {
        var i0 = 0;
        for (var iy = 0; iy < n; iy++) {
            for (var ix = 0; ix < n - iy; ix++) {
                v_re[i0 + ix] = dot(b, vec3f(v_re[i0 + ix], v_re[i0 + ix + 1], v_re[i0 + ix + dy - iy]));
            }
            i0 += dy - iy;
        }
    }

    if icomp == -1 {
        // For norm mode, result is already real
        return v_re[0];
    }

    // De Casteljau on imaginary part
    for (var n = order; n > 0; n--) {
        var i0 = 0;
        for (var iy = 0; iy < n; iy++) {
            for (var ix = 0; ix < n - iy; ix++) {
                v_im[i0 + ix] = dot(b, vec3f(v_im[i0 + ix], v_im[i0 + ix + 1], v_im[i0 + ix + dy - iy]));
            }
            i0 += dy - iy;
        }
    }

    let re = v_re[0];
    let im = v_im[0];

    // Combine based on complex mode
    switch u_complex.mode {
        case 0u: {
            // Phase rotation: Re(z * e^{iφ})
            let c = cos(u_complex.phase);
            let s = sin(u_complex.phase);
            return re * c - im * s;
        }
        case 1u: { return sqrt(re * re + im * im); }  // Abs
        case 2u: { return atan2(im, re); }             // Arg
        default: {
            let c = cos(u_complex.phase);
            let s = sin(u_complex.phase);
            return re * c - im * s;
        }
    }
}

fn _evalTrigVec3Data(order: i32, data: array<vec3f, N_DOFS_TRIG_VEC3>, lam: vec2f, dy: i32) -> vec3f {
    let b = vec3f(lam.x, lam.y, 1.0 - lam.x - lam.y);
    var v: array<vec3f, N_DOFS_TRIG_VEC3> = data;

    for (var n = order; n > 0; n--) {
        var i0 = 0;
        for (var iy = 0; iy < n; iy++) {
            for (var ix = 0; ix < n - iy; ix++) {
                v[i0 + ix] = mat3x3<f32>(v[i0 + ix], v[i0 + ix + 1], v[i0 + ix + dy - iy]) * b;
            }
            i0 += dy - iy;
        }
    }

    return v[0];
}

fn evalTrigVec3(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>, offset_: u32) -> vec3f {
    var order: i32 = i32((*data)[offset_ + 1]);
    let ncomp: u32 = u32((*data)[offset_ + 0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    var v: array<vec3f, N_DOFS_TRIG_VEC3>;
    let offset: u32 = offset_ + ndof * id * ncomp + VALUES_OFFSET;
    let stride: u32 = ncomp;

    for (var i: u32 = 0u; i < ndof; i++) {
        v[i].x = (*data)[offset + i * stride + 0];
        v[i].y = (*data)[offset + i * stride + 1];
        v[i].z = (*data)[offset + i * stride + 2];
    }

    return _evalTrigVec3Data(order, v, lam, order+1);
}

fn evalTrigVec3Grad(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>, offset_: u32) -> mat3x3<f32> {
    var order: i32 = i32((*data)[offset_ + 1]);
    let ncomp: u32 = u32((*data)[offset_ + 0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);
    let dy = order + 1;

    var v: array<vec3f, N_DOFS_TRIG_VEC3>;
    let offset: u32 = offset_ + ndof * id * ncomp + VALUES_OFFSET;
    let stride: u32 = ncomp;

    for (var i: u32 = 0u; i < ndof; i++) {
        v[i].x = (*data)[offset + i * stride + 0];
        v[i].y = (*data)[offset + i * stride + 1];
        v[i].z = (*data)[offset + i * stride + 2];
    }

    var result: mat3x3<f32>;

    result[0] = _evalTrigVec3Data(order, v, lam, dy);
    var vd: array<vec3f, N_DOFS_TRIG_VEC3>;

    var i0 = 0;
    for (var iy = 0; iy < order; iy++) {
        for (var ix = 0; ix < order - iy; ix++) {
            vd[i0 + ix] = v[i0 + ix] - v[i0 + ix + dy - iy];
        }
        i0 += dy - iy;
    }

    result[1] = _evalTrigVec3Data(order-1, vd, lam, dy);

    i0 = 0;
    for (var iy = 0; iy < order; iy++) {
        for (var ix = 0; ix < order - iy; ix++) {
            vd[i0 + ix] = v[i0 + ix + 1] - v[i0 + ix + dy - iy];
        }
        i0 += dy - iy;
    }

    result[2] = _evalTrigVec3Data(order-1, vd, lam, dy);

    return result;
}
