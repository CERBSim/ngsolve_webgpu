#import clipping
// #import colormap
#import ngsolve/clipping/common

@group(0) @binding(21) var<storage, read_write> count_vectors: atomic<u32>;
@group(0) @binding(22) var<storage, read_write> positions: array<f32>;
@group(0) @binding(23) var<storage, read_write> directions: array<f32>;
#ifdef IS_COMPLEX
@group(0) @binding(25) var<storage, read_write> directions_imag: array<f32>;
#endif IS_COMPLEX
@group(0) @binding(29) var<storage, read_write> values: array<f32>;
@group(0) @binding(24) var<uniform> u_ntets: u32;
@group(0) @binding(31) var<uniform> u_gridsize: f32;

#ifdef NEED_EVAL
fn evalTetReIm(data: ptr<storage, array<f32>, read>,
               id: u32, icomp: i32, lam: vec3<f32>) -> vec2f {
  let ncomp: u32 = u32((*data)[0]);
  let order: u32 = u32((*data)[1]);
  let is_complex: u32 = u32((*data)[2]);
  var ndof: u32 = ((order + 1u) * (order + 2u) * (order + 3u)) / 6u;

  let stride: u32 = ncomp * (1u + is_complex);
  let offset: u32 = ndof * id * stride + VALUES_OFFSET;
  var lam_res = clamp(vec4f(lam.xyz, 1.0 - lam.x - lam.y - lam.z),
                      vec4f(0.0), vec4f(1.0));
  lam_res *= 1.0 / (lam_res.x + lam_res.y + lam_res.z + lam_res.w);

  var re_val: f32 = 0.0;
  var im_val: f32 = 0.0;
  let comp = u32(icomp);

  if is_complex == 0u {
    var j: u32 = 0u;
    for(var d: u32 = 0u; d < order+1u; d++) {
      for(var c: u32 = 0u; c < order+1u-d; c++) {
        for(var b: u32 = 0u; b < order+1u-c-d; b++) {
          let a = order - b - c - d;
          let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
          let basis = fac * mypow(lam_res.x, a) * mypow(lam_res.y, b) * mypow(lam_res.z, c) * mypow(lam_res.w, d);
          re_val += basis * (*data)[offset + comp + j * stride];
          j++;
        }
      }
    }
    return vec2f(re_val, 0.0);
  }

  // Complex path
  let comp_offset = comp * 2u;
  var j: u32 = 0u;
  for(var d: u32 = 0u; d < order+1u; d++) {
    for(var c: u32 = 0u; c < order+1u-d; c++) {
      for(var b: u32 = 0u; b < order+1u-c-d; b++) {
        let a = order - b - c - d;
        let fac = f32(factorial(order))/f32((factorial(a) * factorial(b) * factorial(c) * factorial(d)));
        let basis = fac * mypow(lam_res.x, a) * mypow(lam_res.y, b) * mypow(lam_res.z, c) * mypow(lam_res.w, d);
        re_val += basis * (*data)[offset + j * stride + comp_offset];
        im_val += basis * (*data)[offset + j * stride + comp_offset + 1u];
        j++;
      }
    }
  }
  return vec2f(re_val, im_val);
}
#endif NEED_EVAL

@compute @workgroup_size(256)
fn compute_clipping_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
  for (var tetId = id.x; tetId<u_ntets; tetId+=256*1024) {
    let p_tet = getTetPoints(tetId);
    let lam_clip = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));

    let f = array(dot(vec4<f32>(p_tet[0], 1.0), u_clipping.plane),
                  dot(vec4<f32>(p_tet[1], 1.0), u_clipping.plane),
                  dot(vec4<f32>(p_tet[2], 1.0), u_clipping.plane),
                  dot(vec4<f32>(p_tet[3], 1.0), u_clipping.plane));
    let cuts = clipTet(lam_clip, f, tetId);
    if(cuts.n == 0) {
      continue;
    }
    // let index = atomicAdd(&count_vectors, cuts.n);
    for(var iCut = 0u; iCut < cuts.n; iCut++) {
      let lam_trig = cuts.trigs[iCut].lam;

      var p = array<vec3<f32>, 3>();

      for(var pi = 0u; pi < 3; pi++) {
        p[pi] = vec3<f32>(0.0, 0.0, 0.0);
        let lam = lam_trig[pi];
        for(var j = 0u; j < 3u; j++) {
          p[pi] += lam[j] * p_tet[j];
        }
        let lam4 = 1.0 - lam.x - lam.y - lam.z;
        p[pi] += lam4 * p_tet[3];
      }

    let pmin = vec3f(-0.12320326, -0.185135, -0.185136);
    let rad = 1.0;

    var dir: u32 =0;
    var dir1: u32 =0;
    var dir2: u32 =0;

    let n = cross (p[1]-p[0], p[2]-p[0]);
    let na  = abs(n);
    if (na[0] > na[1] && na[0] > na[2]) {
      dir = 0;
    }
    else if (na[1] > na[2]) {
      dir = 1;
    }
    else  {
      dir = 2;
    }
    
    dir1 = (dir+1) % 3;
    dir2 = (dir1+1) % 3;

    var p2d: array<vec2f, 3>;

    for (var k: u32 = 0; k < 3; k++)
      {
        p2d[k] = vec2f((p[k][dir1] - pmin[dir1]) / (2*rad),
                       (p[k][dir2] - pmin[dir2]) / (2*rad));
      }

    var min2d = min(min(p2d[0], p2d[1]), p2d[2]);
    var max2d = max(max(p2d[0], p2d[1]), p2d[2]);

    let m = mat2x2f(
      p2d[1] - p2d[0],
      p2d[2] - p2d[0]
    );
    let mdet = determinant(m);

    let minv = 1.0/mdet * mat2x2f( m[1][1], -m[0][1], -m[1][0], m[0][0] );
    
    let gridsize = u_gridsize;
    let smin = ceil(min2d.x / gridsize) * gridsize;
    let smax = floor(max2d.x / gridsize) * gridsize;
    let tmin = ceil(min2d.y / gridsize) * gridsize;
    let tmax = floor(max2d.y / gridsize) * gridsize;

    for (var s = smin; s <= smax+1e-6; s += gridsize) {
        for (var t = tmin; t <= tmax+1e-6; t += gridsize) {
              let lam = minv * (vec2f(s, t) - p2d[0]);
          /*
              var lam = vec2f(0.0);
              if(dir == 0) {
                lam = vec2f(1.0-lam2d.x-lam2d.y, lam2d.x);
              }
              if(dir == 1) {
                lam = vec2f(lam.y, 1.0-lam2d.x-lam2d.y);
              }
              if(dir == 2) {
                lam = vec2f(lam.x, lam.y);
              }
              
          */
              if (lam.x >= 0 && lam.y >= 0 && lam.x+lam.y <= 1.0)
                {
                  var cp = p[0] + lam.x * (p[1] - p[0]) + lam.y * (p[2] - p[0]);

#ifdef NEED_EVAL
                      // write output to buffer
                      let lamTet = lam_trig[0] + lam.x * (lam_trig[1]-lam_trig[0]) + lam.y * (lam_trig[2] - lam_trig[0]);

                      var v_re = vec3f(0.0);
                      var v_im = vec3f(0.0);
                      for(var j = 0; j < 3; j++) {
                        let ri = evalTetReIm(&u_function_values_3d, tetId, j, lamTet);
                        v_re[j] = ri.x;
                        v_im[j] = ri.y;
                      }
                      let val = sqrt(dot(v_re, v_re) + dot(v_im, v_im));

#ifdef SCALE_BY_VALUE
                      let dir_re = v_re;
                      let dir_im = v_im;
#else SCALE_BY_VALUE
                      let scale = 2 * gridsize * 1.0;
                      let dir_re = scale * v_re / max(val, 1e-10);
                      let dir_im = scale * v_im / max(val, 1e-10);
#endif SCALE_BY_VALUE
                      let index = atomicAdd(&count_vectors, 1);
                      cp += 0.5 * gridsize * normalize(cross(p[1]-p[0], p[2]-p[0]));

                      positions[index*3+0] = cp[0];
                      positions[index*3+1] = cp[1];
                      positions[index*3+2] = cp[2];
                      values[index] = val;
                      directions[index*3+0] = dir_re[0];
                      directions[index*3+1] = dir_re[1];
                      directions[index*3+2] = dir_re[2];
#ifdef IS_COMPLEX
                      directions_imag[index*3+0] = dir_im[0];
                      directions_imag[index*3+1] = dir_im[1];
                      directions_imag[index*3+2] = dir_im[2];
#endif IS_COMPLEX
#else NEED_EVAL
                      atomicAdd(&count_vectors, 1);
#endif NEED_EVAL

                }


  }
    }
      }
}
}
