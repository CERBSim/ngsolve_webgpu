#import ngsolve/eval/common

const N_DOFS_TET: u32 = (@MAX_EVAL_ORDER@+1) * (@MAX_EVAL_ORDER@ + 2) * (@MAX_EVAL_ORDER@ + 3) / 6;
const N_DOFS_TET_VEC3: u32 = (@MAX_EVAL_ORDER_VEC3@+1) * (@MAX_EVAL_ORDER_VEC3@ + 2) * (@MAX_EVAL_ORDER_VEC3@ + 3) / 6;

// De Casteljau evaluation of vec3-valued Bernstein polynomial on a tet.
// Data layout: coefficients stored in packed tet order (ix innermost, iy middle, iz outermost).
// lam = (x, y, z) reference tet coordinates.
// Barycentric: b0=lam.x, b1=lam.y, b2=lam.z, b3=1-x-y-z
fn _evalTetVec3DeCasteljau(v: ptr<function, array<vec3f, N_DOFS_TET_VEC3>>, order: i32, lam: vec3f) -> vec3f {
    let b = vec4f(lam, 1.0 - lam.x - lam.y - lam.z);

    for (var n = order; n > 0; n--) {
        var idx = 0;
        for (var iz = 0; iz < n; iz++) {
            for (var iy = 0; iy < n - iz; iy++) {
                let stride_y = order + 1 - iz - iy;
                let stride_z = (order - iz + 1) * (order - iz + 2) / 2 - iy;

                for (var ix = 0; ix < n - iz - iy; ix++) {
                    let p = idx + ix;
                    (*v)[p] = b.x * (*v)[p] + b.y * (*v)[p + 1]
                            + b.z * (*v)[p + stride_y] + b.w * (*v)[p + stride_z];
                }
                idx += order + 1 - iz - iy;
            }
            // skip unprocessed rows in this layer
            for (var iy = n - iz; iy <= order - iz; iy++) {
                idx += order + 1 - iz - iy;
            }
        }
    }

    return (*v)[0];
}

fn evalTetVec3(data: ptr<storage, array<f32>, read>, local_id: u32, lam: vec3f, offset_: u32) -> vec3f {
    let order = bitcast<i32>((*data)[offset_]);
    let n_curved = bitcast<i32>((*data)[offset_ + 1u]);
    let ndof = u32((order + 1) * (order + 2) * (order + 3) / 6);
    // Skip header (2) + lookup table (n_total inferred from caller)
    // local_id is the curved element local index
    // Caller provides offset_ pointing to start of curvature_3d block
    // Layout: [order, n_curved, lookup[n_total], coeffs[n_curved * ndof * 3]]
    // The caller already resolved the lookup; local_id is the curved index.
    // coeff_offset points to the start of this element's coefficients.
    let coeff_offset = offset_ + 2u + u32(n_curved) * 0u; // placeholder, see evalTetVec3At

    var v: array<vec3f, N_DOFS_TET_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_offset + local_id * ndof * 3u + i * 3u;
        v[i] = vec3f((*data)[base], (*data)[base + 1u], (*data)[base + 2u]);
    }

    return _evalTetVec3DeCasteljau(&v, order, lam);
}

// Evaluate tet vec3 Bernstein at given element, reading from mesh.data
// n_total: total number of 3D elements (needed to skip lookup table)
// local_id: curved element local index (from lookup)
fn evalTetVec3At(offset_: u32, n_total: u32, local_id: u32, ndof: u32, lam: vec3f) -> vec3f {
    let order = bitcast<i32>(mesh.data[offset_]);
    let coeff_base = offset_ + 2u + n_total + local_id * ndof * 3u;

    var v: array<vec3f, N_DOFS_TET_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_base + i * 3u;
        v[i] = vec3f(mesh.data[base], mesh.data[base + 1u], mesh.data[base + 2u]);
    }

    return _evalTetVec3DeCasteljau(&v, order, lam);
}

// Gradient: returns [position, dF/dx, dF/dy, dF/dz] packed as mat4x3
// Uses derivative of Bernstein = order * differences, evaluated via de Casteljau at order-1
fn evalTetVec3GradAt(offset_: u32, n_total: u32, local_id: u32, ndof: u32, lam: vec3f) -> mat4x3f {
    let order = bitcast<i32>(mesh.data[offset_]);
    let coeff_base = offset_ + 2u + n_total + local_id * ndof * 3u;

    var v: array<vec3f, N_DOFS_TET_VEC3>;
    for (var i = 0u; i < ndof; i++) {
        let base = coeff_base + i * 3u;
        v[i] = vec3f(mesh.data[base], mesh.data[base + 1u], mesh.data[base + 2u]);
    }

    // Evaluate position
    var vp = v;
    let pos = _evalTetVec3DeCasteljau(&vp, order, lam);

    // Derivative in x-direction (lam.x): d/dx B = order * (B_{a-1,b+1,c,d} - B_{a-1,b,c,d+1})
    // In terms of our indexing: dF/dx = order * (v[ix+1,iy,iz] - v[ix,iy,iz+1])
    // These differences form a degree order-1 polynomial
    var vdx: array<vec3f, N_DOFS_TET_VEC3>;
    var vdy: array<vec3f, N_DOFS_TET_VEC3>;
    var vdz: array<vec3f, N_DOFS_TET_VEC3>;

    var src_idx = 0;
    var dst_idx = 0;
    for (var iz = 0; iz < order; iz++) {
        for (var iy = 0; iy < order - iz; iy++) {
            let stride_y = order + 1 - iz - iy;
            let stride_z = (order - iz + 1) * (order - iz + 2) / 2 - iy;

            for (var ix = 0; ix < order - iz - iy; ix++) {
                let p = src_idx + ix;
                // dP/dx: v[ix,iy,iz] - v[ix,iy,iz+1]
                vdx[dst_idx] = v[p] - v[p + stride_z];
                // dP/dy: v[ix+1,iy,iz] - v[ix,iy,iz+1]
                vdy[dst_idx] = v[p + 1] - v[p + stride_z];
                // dP/dz: v[ix,iy+1,iz] - v[ix,iy,iz+1]
                vdz[dst_idx] = v[p + stride_y] - v[p + stride_z];
                dst_idx++;
            }
            src_idx += order + 1 - iz - iy;
        }
        // skip remaining rows
        for (var iy = order - iz; iy <= order - iz; iy++) {
            src_idx += order + 1 - iz - iy;
        }
    }

    let fo = f32(order);
    let dx = _evalTetVec3DeCasteljau(&vdx, order - 1, lam) * fo;
    let dy = _evalTetVec3DeCasteljau(&vdy, order - 1, lam) * fo;
    let dz = _evalTetVec3DeCasteljau(&vdz, order - 1, lam) * fo;

    return mat4x3f(pos, dx, dy, dz);
}

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
