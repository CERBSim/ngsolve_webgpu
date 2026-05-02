#import ngsolve/mesh/utils

@group(0) @binding(21) var<storage, read_write> count_vectors: atomic<u32>;
@group(0) @binding(31) var<uniform> u_gridsize: f32;

@compute @workgroup_size(256)
fn compute_surface_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
  let n_trigs = mesh.num_trigs;
  for (var trigId = id.x; trigId<n_trigs; trigId+=256*1024) {
    let p = loadTriangle(trigId).p;
    
    let gridsize = u_gridsize;
    let pmin = vec3f(-1, -1, -1);
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
    
    for (var s = 0.0; s <= 1.; s += 1.0 * gridsize) {
      if (s >= min2d.x && s <= max2d.x) 
      {
        for (var t = 0.; t <= 1.; t += 1.0 * gridsize) {
          if (t >= min2d.y && t <= max2d.y)
            {
              let lam = minv * (vec2f(s, t) - p2d[0]);
              
              if (lam.x >= 0 && lam.y >= 0 && lam.x+lam.y <= 1)
                {
                  atomicAdd(&count_vectors, 1);
                }
  }
    }
      }
    }
  }
}
