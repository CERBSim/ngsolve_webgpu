#import clipping
#import ngsolve/clipping/common
#import ngsolve/eval/tet

@group(0) @binding(21) var<storage, read_write> count_trigs: atomic<u32>;
@group(0) @binding(22) var<uniform> u_ntets: u32;
@group(0) @binding(23) var<uniform> only_count: u32;
@group(0) @binding(24) var<storage, read_write> subtrigs: array<SubTrig>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    
  let offset_3d = mesh.offset_3d_data;
  let n_tets = bitcast<u32>(mesh.data[offset_3d + 0]) + bitcast<u32>(mesh.data[offset_3d + 2]) + 2 * bitcast<u32>(mesh.data[offset_3d + 3]) + 5*bitcast<u32>(mesh.data[offset_3d + 4]);
  
  for (var i = id.x; i<n_tets; i+=256*1024) {
    let p = getTetPoints(i);
    let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));

    let f = array(dot(vec4<f32>(p[0], 1.0), u_clipping.plane),
                  dot(vec4<f32>(p[1], 1.0), u_clipping.plane),
                  dot(vec4<f32>(p[2], 1.0), u_clipping.plane),
                  dot(vec4<f32>(p[3], 1.0), u_clipping.plane));
    let cuts = clipTet(lam, f, i);
    if(cuts.n == 0) {
      continue;
    }
    let index = atomicAdd(&count_trigs, cuts.n);
    if(only_count == u32(1)) {
      continue;
    }
    for(var k = 0u; k < cuts.n; k++) {
      subtrigs[index+k] = cuts.trigs[k];
    }
  }
}

