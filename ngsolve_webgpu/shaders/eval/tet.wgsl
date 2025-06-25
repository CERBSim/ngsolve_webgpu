#import ngsolve/eval/common

fn factorial(n: u32) -> u32 {
    var result: u32 = 1u;
    var i: u32 = 2u;
    while (i <= n) {
        result = result * i;
        i = i + 1u;
    }
    return result;
}

fn mypow(val: f32, exp: u32) -> f32 {
    var result: f32 = 1.0;
    for (var i: u32 = 0u; i < exp; i = i + 1u) {
        result = result * val;
    }
    return result;
}

fn evalTet(data: ptr<storage, array<f32>, read>,
           id: u32, icomp: i32, lam: vec3<f32>) -> f32 {
  let ncomp: u32 = u32((*data)[0]);
  let order: u32 = u32((*data)[1]);
  var ndof: u32 = ((order + 1u) * (order + 2u) * (order + 3u)) / 6u;

  let offset: u32 = ndof * id * ncomp + VALUES_OFFSET;
  let stride: u32 = ncomp;
  var lam_res = clamp(vec4f(lam.xyz, 1.0 - lam.x - lam.y - lam.z),
                      vec4f(0.0, 0.0, 0.0, 0.0),
                      vec4f(1.0, 1.0, 1.0, 1.0));
  lam_res *= 1.0 / (lam_res.x + lam_res.y + lam_res.z + lam_res.w);

  var value: f32 = 0.0;
  var j: u32 = 0u;
  for(var d: u32 = 0u; d < order+1u; d++) {
    for(var c: u32 = 0u; c < order+1u-d; c++) {
      for(var b: u32 = 0u; b < order+1u-c-d;b++) {
        let a = order - b - c - d;
        let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
        if (icomp == -1)
          {
            var comp_val = 0.;
            for(var k: u32 = 0u; k < ncomp; k++) {
              comp_val = comp_val + mypow(fac * (*data)[offset + j] * mypow(lam_res.x, a) * mypow(lam_res.y, b) * mypow(lam_res.z, c) * mypow(lam_res.w, d), 2u);
              j++;
            }
            value = value + sqrt(comp_val);
          }
        else
          {
            value = value + fac * (*data)[offset + u32(icomp) + j * stride] * mypow(lam.x, a) * mypow(lam.y, b) * mypow(lam.z, c) * mypow(1.0 - lam.x - lam.y - lam.z, d);
            j++;
          }
      }
    }
  }
  return value;
    // let dy = order + 1u;
    // let dz = (order + 1u) * (order + 2u) / 2u;
    // let b = vec4f(lam.x, lam.y, lam.z, 1.0 - lam.x - lam.y - lam.z);

    // for (var n = order; n > 0u; n--) {
    //     var iz0 = 0u;
    //     for (var iz = 0u; iz < n; iz++) {
    //         var iy0 = iz0;
    //         for (var iy = 0u; iy < n - iz; iy++) {
    //             for (var ix = 0u; ix < n - iz - iy; ix++) {
    //                 v[iy0 + ix] = dot(b, vec4f(v[iy0 + ix], v[iy0 + ix + 1u], v[iy0 + ix + dy - iy], v[iy0 + ix + dz - iz]));
    //             }
    //             iy0 += dy - iy - iz;
    //         }
    //         iz0 += dz - (n - 1u - iz);
    //     }
    // }

    // return v[0];
}
