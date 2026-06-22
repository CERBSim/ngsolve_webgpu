#import ngsolve/eval/trig
#import ngsolve/mesh/utils

@group(0) @binding(21) var<storage, read_write> count_vectors: atomic<u32>;
@group(0) @binding(22) var<storage, read_write> positions: array<f32>;
@group(0) @binding(23) var<storage, read_write> directions: array<f32>;
#ifdef IS_COMPLEX
@group(0) @binding(25) var<storage, read_write> directions_imag: array<f32>;
#endif IS_COMPLEX
@group(0) @binding(29) var<storage, read_write> values: array<f32>;
@group(0) @binding(31) var<uniform> u_gridsize: f32;

@compute @workgroup_size(256)
fn compute_surface_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
  let n_trigs = mesh.num_trigs;
  for (var trigId = id.x; trigId<n_trigs; trigId+=256*1024) {
    var p = loadTriangle(trigId).p;
    
    if (mesh.is_curved == 1u) {
      p[0] = evalTrigVec3(&mesh.data, trigId, vec2f(1.0, 0.0), mesh.offset_curvature_2d);
      p[1] = evalTrigVec3(&mesh.data, trigId, vec2f(0.0, 1.0), mesh.offset_curvature_2d);
      p[2] = evalTrigVec3(&mesh.data, trigId, vec2f(0.0, 0.0), mesh.offset_curvature_2d);
    }

    let gridsize = 2.0 * u_gridsize;

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
        p2d[k] = vec2f(p[k][dir1], p[k][dir2]);
      }

    var min2d = min(min(p2d[0], p2d[1]), p2d[2]);
    var max2d = max(max(p2d[0], p2d[1]), p2d[2]);

    let m = mat2x2f(
      p2d[1] - p2d[0],
      p2d[2] - p2d[0]
    );
    let mdet = determinant(m);

    let minv = 1.0/mdet * mat2x2f( m[1][1], -m[0][1], -m[1][0], m[0][0] );

    let s_start = ceil(min2d.x / gridsize) * gridsize;
    let t_start = ceil(min2d.y / gridsize) * gridsize;

    for (var s = s_start; s <= max2d.x; s += gridsize) {
        for (var t = t_start; t <= max2d.y; t += gridsize) {
              let lam = minv * (vec2f(s, t) - p2d[0]);
              
              if (lam.x >= 0 && lam.y >= 0 && lam.x+lam.y <= 1)
                {
                  var cp = p[0] + lam.x * (p[1] - p[0]) + lam.y * (p[2] - p[0]);
                  let eval_lam = vec2f(1.0 - lam.x - lam.y, lam.x);
                  
                  let v_ri = evalTrigVec3ReIm(&u_function_values_2d, trigId, eval_lam, 0u);
                  let val = sqrt(dot(v_ri.re, v_ri.re) + dot(v_ri.im, v_ri.im));

#ifdef SCALE_BY_VALUE
                  let dir_re = v_ri.re;
                  let dir_im = v_ri.im;
#else SCALE_BY_VALUE
                  let dir_re = v_ri.re / max(val, 1e-10);
                  let dir_im = v_ri.im / max(val, 1e-10);
#endif SCALE_BY_VALUE
                  let index = atomicAdd(&count_vectors, 1);
                  if (index * 3u + 2u >= arrayLength(&positions)) { continue; }

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
                }
        }
    }
  }
}
