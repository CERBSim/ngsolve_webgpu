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
  let is_complex: u32 = u32((*data)[2]);
  var ndof: u32 = ((order + 1u) * (order + 2u) * (order + 3u)) / 6u;

  let stride: u32 = ncomp * (1u + is_complex);
  let offset: u32 = ndof * id * stride + VALUES_OFFSET;

  if is_complex == 0u {
    // Real path
    var value : f32 = 0.0;
    var first_comp = 0u;
    if(icomp != -1) {
      first_comp = u32(icomp);
    }
    var last_comp = ncomp;
    if(icomp != -1) {
      last_comp = u32(icomp) + 1u;
    }

    for(var jcomp: u32 = first_comp; jcomp < last_comp; jcomp++) {
      var j: u32 = 0u;
      var comp_value = 0.0;
      for(var d: u32 = 0u; d < order+1u; d++) {
        for(var c: u32 = 0u; c < order+1u-d; c++) {
          for(var b: u32 = 0u; b < order+1u-c-d; b++) {
            let a = order - b - c - d;
            let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
            comp_value = comp_value + fac * (*data)[offset + jcomp + j * stride] * mypow(lam.x, a) * mypow(lam.y, b) * mypow(lam.z, c) * mypow(1.0 - lam.x - lam.y - lam.z, d);
            j++;
          }
        }
      }
      if(icomp != -1) {
        return comp_value;
      }
      value = value + comp_value*comp_value;
    }
    return sqrt(value);
  }

  // Complex path
  if icomp == -1 {
    // Norm: sqrt(sum_k(re_k^2 + im_k^2)) per DOF, then Bernstein eval
    var value = 0.0;
    var j: u32 = 0u;
    for(var d: u32 = 0u; d < order+1u; d++) {
      for(var c: u32 = 0u; c < order+1u-d; c++) {
        for(var b: u32 = 0u; b < order+1u-c-d; b++) {
          let a = order - b - c - d;
          let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
          var dof_norm = 0.0;
          for (var k: u32 = 0u; k < ncomp; k++) {
            let re = (*data)[offset + j * stride + k * 2u];
            let im = (*data)[offset + j * stride + k * 2u + 1u];
            dof_norm += re * re + im * im;
          }
          dof_norm = sqrt(dof_norm);
          value += fac * dof_norm * mypow(lam.x, a) * mypow(lam.y, b) * mypow(lam.z, c) * mypow(1.0 - lam.x - lam.y - lam.z, d);
          j++;
        }
      }
    }
    return value;
  }

  // Specific component: evaluate Re and Im separately
  let comp_offset = u32(icomp) * 2u;
  var re_value = 0.0;
  var im_value = 0.0;
  var j: u32 = 0u;
  for(var d: u32 = 0u; d < order+1u; d++) {
    for(var c: u32 = 0u; c < order+1u-d; c++) {
      for(var b: u32 = 0u; b < order+1u-c-d; b++) {
        let a = order - b - c - d;
        let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
        let basis = fac * mypow(lam.x, a) * mypow(lam.y, b) * mypow(lam.z, c) * mypow(1.0 - lam.x - lam.y - lam.z, d);
        re_value += (*data)[offset + j * stride + comp_offset] * basis;
        im_value += (*data)[offset + j * stride + comp_offset + 1u] * basis;
        j++;
      }
    }
  }

  switch u_complex.mode {
    case 0u: {
      let co = cos(u_complex.phase);
      let si = sin(u_complex.phase);
      return re_value * co - im_value * si;
    }
    case 1u: { return sqrt(re_value * re_value + im_value * im_value); }
    case 2u: { return atan2(im_value, re_value); }
    default: {
      let co = cos(u_complex.phase);
      let si = sin(u_complex.phase);
      return re_value * co - im_value * si;
    }
  }
}

// Complex-aware evalTet for deformation: always uses phase rotation (ignores mode)
fn evalTetComplex(data: ptr<storage, array<f32>, read>,
           id: u32, icomp: i32, lam: vec3<f32>) -> f32 {
  let ncomp: u32 = u32((*data)[0]);
  let order: u32 = u32((*data)[1]);
  let is_complex: u32 = u32((*data)[2]);

  if is_complex == 0u {
    return evalTet(data, id, icomp, lam);
  }

  var ndof: u32 = ((order + 1u) * (order + 2u) * (order + 3u)) / 6u;
  let stride: u32 = ncomp * 2u;
  let offset: u32 = ndof * id * stride + VALUES_OFFSET;
  let comp_offset = u32(icomp) * 2u;

  var re_value = 0.0;
  var im_value = 0.0;
  var j: u32 = 0u;
  for(var d: u32 = 0u; d < order+1u; d++) {
    for(var c: u32 = 0u; c < order+1u-d; c++) {
      for(var b: u32 = 0u; b < order+1u-c-d; b++) {
        let a = order - b - c - d;
        let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
        let basis = fac * mypow(lam.x, a) * mypow(lam.y, b) * mypow(lam.z, c) * mypow(1.0 - lam.x - lam.y - lam.z, d);
        re_value += (*data)[offset + j * stride + comp_offset] * basis;
        im_value += (*data)[offset + j * stride + comp_offset + 1u] * basis;
        j++;
      }
    }
  }

  let co = cos(u_complex.phase);
  let si = sin(u_complex.phase);
  return re_value * co - im_value * si;
}
