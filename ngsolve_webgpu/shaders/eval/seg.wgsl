#import ngsolve/eval/common

fn evalSeg(data: ptr<storage, array<f32>, read>, id: u32, icomp: u32, lam: f32) -> f32 {
    let order: u32 = u32((*data)[1]);
    let ncomp: u32 = u32((*data)[0]);
    let ndof: u32 = order + 1;

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    var v: array<f32, 7>;
    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = (*data)[offset + i * stride];
    }

    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = (*data)[offset + i * stride];
    }

    let b = vec2f(lam, 1.0 - lam);

    for (var n = order; n > 0; n--) {
        for (var i = 0u; i < n; i++) {
            v[i] = dot(b, vec2f(v[i], v[i + 1]));
        }
    }

    return v[0];
}

fn evalSegVec3(data: ptr<storage, array<f32>, read>, id: u32, lam: f32, offset_field: u32) -> vec3<f32> {
    let order: u32 = u32((*data)[1]);
    let ndof: u32 = order + 1;

    let offset: u32 = offset_field + id * ndof * 3;

    var v: array<vec3<f32>, 7>;
    for (var i: u32 = 0u; i < ndof; i++) {
        let base = offset + i * 3;
        v[i] = vec3<f32>((*data)[base], (*data)[base + 1], (*data)[base + 2]);
    }

    let b = vec2f(lam, 1.0 - lam);

    for (var n = order; n > 0; n--) {
        for (var i = 0u; i < n; i++) {
            v[i] = b.x * v[i] + b.y * v[i + 1];
        }
    }

    return v[0];
}

