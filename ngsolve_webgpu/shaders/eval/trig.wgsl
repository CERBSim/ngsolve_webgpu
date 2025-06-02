#import ngsolve/eval/common

const N_DOFS_TRIG: u32 = (@MAX_EVAL_ORDER@+1) * (@MAX_EVAL_ORDER@ + 2) / 2;
const N_DOFS_TRIG_VEC3: u32 = (@MAX_EVAL_ORDER_VEC3@+1) * (@MAX_EVAL_ORDER_VEC3@ + 2) / 2;

fn evalTrig(data: ptr<storage, array<f32>, read>, id: u32, icomp: i32, lam: vec2<f32>) -> f32 {
    var order: i32 = i32((*data)[1]);
    let ncomp: u32 = u32((*data)[0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    var v: array<f32, N_DOFS_TRIG>;
    let offset: u32 = ndof * id * ncomp + VALUES_OFFSET;
    let stride: u32 = ncomp;

    if icomp == -1 {
        // norm of vector
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

fn evalTrigVec3(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>) -> vec3f {
    var order: i32 = i32((*data)[1]);
    let ncomp: u32 = u32((*data)[0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    var v: array<vec3f, N_DOFS_TRIG_VEC3>;
    let offset: u32 = ndof * id * ncomp + VALUES_OFFSET;
    let stride: u32 = ncomp;

    for (var i: u32 = 0u; i < ndof; i++) {
        v[i].x = (*data)[offset + i * stride + 0];
        v[i].y = (*data)[offset + i * stride + 1];
        v[i].z = (*data)[offset + i * stride + 2];
    }

    return _evalTrigVec3Data(order, v, lam, order+1);
}

fn evalTrigVec3Grad(data: ptr<storage, array<f32>, read>, id: u32, lam: vec2<f32>) -> mat3x3<f32> {
    var order: i32 = i32((*data)[1]);
    let ncomp: u32 = u32((*data)[0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);
    let dy = order + 1;

    var v: array<vec3f, N_DOFS_TRIG_VEC3>;
    let offset: u32 = ndof * id * ncomp + VALUES_OFFSET;
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

