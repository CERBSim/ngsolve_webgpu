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

struct Vec3ReIm {
    re: vec3f,
    im: vec3f,
};

fn evalTrigVec3ReIm(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>, offset_: u32) -> Vec3ReIm {
    var order: i32 = i32((*data)[offset_ + 1]);
    let ncomp: u32 = u32((*data)[offset_ + 0]);
    let is_complex: u32 = u32((*data)[offset_ + 2]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    let stride: u32 = ncomp * (1u + is_complex);
    let offset: u32 = offset_ + ndof * id * stride + VALUES_OFFSET;

    var v_re: array<vec3f, N_DOFS_TRIG_VEC3>;
    var v_im: array<vec3f, N_DOFS_TRIG_VEC3>;

    if is_complex == 0u {
        for (var i: u32 = 0u; i < ndof; i++) {
            v_re[i].x = (*data)[offset + i * stride + 0];
            v_re[i].y = (*data)[offset + i * stride + 1];
            v_re[i].z = (*data)[offset + i * stride + 2];
            v_im[i] = vec3f(0.0);
        }
    } else {
        for (var i: u32 = 0u; i < ndof; i++) {
            v_re[i].x = (*data)[offset + i * stride + 0];
            v_im[i].x = (*data)[offset + i * stride + 1];
            v_re[i].y = (*data)[offset + i * stride + 2];
            v_im[i].y = (*data)[offset + i * stride + 3];
            v_re[i].z = (*data)[offset + i * stride + 4];
            v_im[i].z = (*data)[offset + i * stride + 5];
        }
    }

    var result: Vec3ReIm;
    result.re = _evalTrigVec3Data(order, v_re, lam, order + 1);
    result.im = _evalTrigVec3Data(order, v_im, lam, order + 1);
    return result;
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

// Complex-aware version of evalTrigVec3 for deformation: always uses phase rotation
fn evalTrigVec3Complex(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>, offset_: u32) -> vec3f {
    let is_complex: u32 = u32((*data)[offset_ + 2]);
    if is_complex == 0u {
        return evalTrigVec3(data, id, lam, offset_);
    }
    let ri = evalTrigVec3ReIm(data, id, lam, offset_);
    let c = cos(u_complex.phase);
    let s = sin(u_complex.phase);
    return ri.re * c - ri.im * s;
}

struct Vec3GradReIm {
    re: mat3x3<f32>,
    im: mat3x3<f32>,
};

fn _evalTrigVec3GradFromData(order: i32, v: array<vec3f, N_DOFS_TRIG_VEC3>, lam: vec2f, dy: i32) -> mat3x3<f32> {
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

// Complex-aware version of evalTrigVec3Grad for deformation: always uses phase rotation
fn evalTrigVec3GradComplex(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>, offset_: u32) -> mat3x3<f32> {
    var order: i32 = i32((*data)[offset_ + 1]);
    let ncomp: u32 = u32((*data)[offset_ + 0]);
    let is_complex: u32 = u32((*data)[offset_ + 2]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);
    let dy = order + 1;

    let stride: u32 = ncomp * (1u + is_complex);
    let offset: u32 = offset_ + ndof * id * stride + VALUES_OFFSET;

    if is_complex == 0u {
        var v: array<vec3f, N_DOFS_TRIG_VEC3>;
        for (var i: u32 = 0u; i < ndof; i++) {
            v[i].x = (*data)[offset + i * stride + 0];
            v[i].y = (*data)[offset + i * stride + 1];
            v[i].z = (*data)[offset + i * stride + 2];
        }
        return _evalTrigVec3GradFromData(order, v, lam, dy);
    }

    var v_re: array<vec3f, N_DOFS_TRIG_VEC3>;
    var v_im: array<vec3f, N_DOFS_TRIG_VEC3>;
    for (var i: u32 = 0u; i < ndof; i++) {
        v_re[i].x = (*data)[offset + i * stride + 0];
        v_im[i].x = (*data)[offset + i * stride + 1];
        v_re[i].y = (*data)[offset + i * stride + 2];
        v_im[i].y = (*data)[offset + i * stride + 3];
        v_re[i].z = (*data)[offset + i * stride + 4];
        v_im[i].z = (*data)[offset + i * stride + 5];
    }

    let grad_re = _evalTrigVec3GradFromData(order, v_re, lam, dy);
    let grad_im = _evalTrigVec3GradFromData(order, v_im, lam, dy);

    let c = cos(u_complex.phase);
    let s = sin(u_complex.phase);
    return mat3x3<f32>(
        grad_re[0] * c - grad_im[0] * s,
        grad_re[1] * c - grad_im[1] * s,
        grad_re[2] * c - grad_im[2] * s,
    );
}
