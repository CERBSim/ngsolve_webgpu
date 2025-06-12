#import ngsolve/eval/trig
#import ngsolve/uniforms
#import colormap

@group(0) @binding(21) var<storage, read_write> count_vectors: atomic<u32>;
@group(0) @binding(22) var<storage, read_write> positions: array<f32>;
@group(0) @binding(23) var<storage, read_write> directions: array<f32>;
@group(0) @binding(25) var<storage, read_write> values: array<f32>;
@group(0) @binding(24) var<uniform> u_ntrigs: u32;

@compute @workgroup_size(256)
fn compute_surface_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
  for (var trigId = id.x; trigId<u_ntrigs; trigId+=256*1024) {
    var vid = 3 * vec3u(
        trigs[4 * trigId + 2],
        trigs[4 * trigId + 0],
        trigs[4 * trigId + 1]
    );

    let p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );

    // let pmin = min(p[0], min(p[1], p[2]));
    let pmin = vec3f(-1, -1, -1);
    let rad = 1.0;
    let swap_lam = false;

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
    
    let gridsize = 0.03;

    let xmin = floor(min2d.x / gridsize) * gridsize;

    for (var s = 0.0; s <= 1.; s += 1.0 * gridsize) {
      if (s >= min2d.x && s <= max2d.x) 
      {
        for (var t = 0.; t <= 1.; t += 1.0 * gridsize) {
          if (t >= min2d.y && t <= max2d.y)
            {
              let lam = minv * (vec2f(s, t) - p2d[0]);
              
              if (lam.x >= 0 && lam.y >= 0 && lam.x+lam.y <= 1)
                {
                  var cp = p[0] + lam.x * (p[1] - p[0]) + lam.y * (p[2] - p[0]);
                  
                  if(@MODE@ == 0) {
  // just count
                      atomicAdd(&count_vectors, 1);
                    }
                    else {
                      // write output to buffer
                      let v = evalTrigVec3(&u_function_values_2d, trigId, lam);
                      
                      let val = length(v);
                      var scale = (val - u_cmap_uniforms.min) / (u_cmap_uniforms.max - u_cmap_uniforms.min);
                      scale = 2 * gridsize * clamp(scale, 0.5, 1.0);
                      let direction = scale * normalize(v) ;
                      let index = atomicAdd(&count_vectors, 1);
                      if (u_curvature_values_2d[0] != -1.) {
                        cp = evalTrigVec3(&u_curvature_values_2d, trigId, lam);
                      }
                      cp += 0.5 * gridsize * normalize(n);

                      positions[index*3+0] = cp[0];
                      positions[index*3+1] = cp[1];
                      positions[index*3+2] = cp[2];
                      values[index] = val;
                      directions[index*3+0] = direction[0];
                      directions[index*3+1] = direction[1];
                      directions[index*3+2] = direction[2];
                    }

                }


  }
    }
      }
    }
  }
}
