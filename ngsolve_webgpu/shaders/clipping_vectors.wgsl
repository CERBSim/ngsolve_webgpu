#import clipping
// #import colormap
#import ngsolve/clipping/common
#import ngsolve/eval/tet

@group(0) @binding(21) var<storage, read_write> count_vectors: atomic<u32>;
@group(0) @binding(22) var<storage, read_write> positions: array<f32>;
@group(0) @binding(23) var<storage, read_write> directions: array<f32>;
@group(0) @binding(29) var<storage, read_write> values: array<f32>;
@group(0) @binding(24) var<uniform> u_ntets: u32;

@compute @workgroup_size(256)
fn compute_clipping_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
  for (var tetId = id.x; tetId<u_ntets; tetId+=256*1024) {
    let p_tet = get_tet_points(tetId);
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
    
    let gridsize = 0.005;
    // let gridsize = 0.5;

    let smin = ceil(min2d.x / gridsize) * gridsize;
    let smax = floor(max2d.x / gridsize) * gridsize;
    let tmin = ceil(min2d.y / gridsize) * gridsize;
    let tmax = floor(max2d.y / gridsize) * gridsize;

    for (var s = smin; s <= smax; s += gridsize) {
        for (var t = tmin; t <= tmax; t += gridsize) {
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
                  
                  if(@MODE@ == 0) {
  // just count
                      atomicAdd(&count_vectors, 1);
                    }
                    else {
                      // write output to buffer
                      var v = vec3f(0.0); 

                      let lamTet = lam_trig[0] + lam.x * (lam_trig[1]-lam_trig[0]) + lam.y * (lam_trig[2] - lam_trig[0]);

                      for(var j = 0; j < 3; j++) {
                        v[j] = evalTet(&u_function_values_3d, tetId, j, lamTet);
                      }
                      
                      let val = length(v);
                      // var scale = (val - u_cmap_uniforms.min) / (u_cmap_uniforms.max - u_cmap_uniforms.min);
                      let scale = 2 * gridsize * 1.0; //clamp(scale, 0.5, 1.0);
                      let direction = scale * normalize(v) ;
                      let index = atomicAdd(&count_vectors, 1);
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
