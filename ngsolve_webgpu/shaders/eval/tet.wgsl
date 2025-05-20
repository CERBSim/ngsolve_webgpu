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
           id: u32, icomp: u32, lam: vec3<f32>) -> f32 {
  let ncomp: u32 = u32((*data)[0]);
  let order: u32 = u32((*data)[1]);
  var ndof: u32 = ((order + 1u) * (order + 2u) * (order + 3u)) / 6u;

  let offset: u32 = ndof * id + VALUES_OFFSET;
  let stride: u32 = ncomp;

  var value: f32 = 0.0;
  var i = 0u;
  for(var d: u32 = 0u; d < order+1u; d++) {
    for(var c: u32 = 0u; c < order+1u-d; c++) {
      for(var b: u32 = 0u; b < order+1u-c-d;b++) {
        let a = order - b - c - d;
        let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
        value = value + fac * (*data)[offset + i * stride] * mypow(lam.x, a) * mypow(lam.y, b) * mypow(lam.z, c) * mypow(1.0 - lam.x - lam.y - lam.z, d);
        i++;
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
