#import ngsolve/mesh/utils

@group(0) @binding(21) var<storage, read_write> count_vectors: atomic<u32>;
@group(0) @binding(31) var<uniform> u_gridsize: f32;

// Inline curved position evaluation using mesh curvature data (de Casteljau).
// Avoids importing eval/trig which would declare extra storage bindings.
fn evalCurvedPos(id: u32, lam: vec2f) -> vec3f {
    let offset_ = mesh.offset_curvature_2d;
    let order = i32(mesh.data[offset_ + 1]);
    let ncomp = u32(mesh.data[offset_ + 0]);
    let ndof = u32((order + 1) * (order + 2) / 2);
    let stride = ncomp;
    let offset = offset_ + ndof * id * stride + 3u;
    let b = vec3f(lam.x, lam.y, 1.0 - lam.x - lam.y);
    let dy = order + 1;

    var w: array<vec3f, 15>;
    for (var i: u32 = 0u; i < min(ndof, 15u); i++) {
        w[i] = vec3f(mesh.data[offset + i * stride],
                     mesh.data[offset + i * stride + 1],
                     mesh.data[offset + i * stride + 2]);
    }

    for (var n = order; n > 0; n--) {
        var i0 = 0;
        for (var iy = 0; iy < n; iy++) {
            for (var ix = 0; ix < n - iy; ix++) {
                w[i0 + ix] = mat3x3<f32>(w[i0 + ix], w[i0 + ix + 1], w[i0 + ix + dy - iy]) * b;
            }
            i0 += dy - iy;
        }
    }
    return w[0];
}

@compute @workgroup_size(256)
fn compute_surface_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
  let n_trigs = mesh.num_trigs;
  for (var trigId = id.x; trigId<n_trigs; trigId+=256*1024) {
    var p = loadTriangle(trigId).p;

    if (mesh.is_curved == 1u) {
      p[0] = evalCurvedPos(trigId, vec2f(1.0, 0.0));
      p[1] = evalCurvedPos(trigId, vec2f(0.0, 1.0));
      p[2] = evalCurvedPos(trigId, vec2f(0.0, 0.0));
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
                  atomicAdd(&count_vectors, 1);
                }
        }
    }
  }
}
