const VALUES_OFFSET: u32 = 2; // storing number of components and order of basis functions in first two entries

fn evalSeg(id: u32, icomp: u32, lam: f32) -> f32 {
    let order: u32 = u32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    let ndof: u32 = order + 1;

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    var v: array<f32, 7>;
    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = seg_function_values[offset + i * stride];
    }

    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = seg_function_values[offset + i * stride];
    }

    let b = vec2f(lam, 1.0 - lam);

    for (var n = order; n > 0; n--) {
        for (var i = 0u; i < n; i++) {
            v[i] = dot(b, vec2f(v[i], v[i + 1]));
        }
    }

    return v[0];
}

fn evalTrig( data: ptr<storage, array<f32>, read>, id: u32, icomp: u32, lam: vec2<f32>) -> f32 {
    var order: i32 = i32((*data)[1]);
    let ncomp: u32 = u32((*data)[0]);
    var ndof: u32 = u32((order + 1) * (order + 2) / 2);

    var v: array<f32, 28>;
    let offset: u32 = ndof * id * ncomp + VALUES_OFFSET;
    let stride: u32 = ncomp;

    if(icomp == 0u)
      {
        // norm of vector
        for (var i: u32 = 0u; i < ndof; i++) {
          v[i] = 0.0;
          for (var j: u32 = 0u; j < ncomp; j++) {
            v[i] += (*data)[offset + i * stride + j] * (*data)[offset + i * stride + j];
          }
          v[i] = sqrt(v[i]);
        }
      }
    else {

        for (var i: u32 = 0u; i < ndof; i++) {
          v[i] = (*data)[offset + icomp-1 + i * stride];
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

fn evalTet(id: u32, icomp: u32, lam: vec3<f32>) -> f32 {
    // Untested (and probably wrong indexing in loop)
    var order: i32 = i32(trig_function_values[1]);
    let ncomp: u32 = u32(trig_function_values[0]);
    var ndof: u32 = u32((order + 1) * (order + 2) * (order + 3) / 6);

    let offset: u32 = ndof * id + VALUES_OFFSET;
    let stride: u32 = ncomp;

    var v: array<f32, 35>; // max order 4
    for (var i: u32 = 0u; i < ndof; i++) {
        v[i] = trig_function_values[offset + i * stride];
    }

    let dy = order + 1;
    let dz = (order + 1) * (order + 2) / 2;
    let b = vec4f(lam.x, lam.y, lam.z, 1.0 - lam.x - lam.y - lam.z);

    for (var n = order; n > 0; n--) {
        var iz0 = 0;
        for (var iz = 0; iz < n; iz++) {
            var iy0 = iz0;
            for (var iy = 0; iy < n - iz; iy++) {
                for (var ix = 0; ix < n - iz - iy; ix++) {
                    v[iy0 + ix] = dot(b, vec4f(v[iy0 + ix], v[iy0 + ix + 1], v[iy0 + ix + dy - iy], v[iy0 + ix + dz - iz]));
                }
                iy0 += dy - iy - iz;
            }
            iz0 += dz - (n - 1 - iz);
        }
    }

    return v[0];
}
