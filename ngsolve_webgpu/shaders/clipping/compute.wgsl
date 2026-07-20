#import clipping
#import ngsolve/clipping/common
#import ngsolve/eval/tet
#import ngsolve/region_visibility

@group(0) @binding(21) var<storage, read_write> count_trigs: atomic<u32>;
@group(0) @binding(22) var<uniform> u_ntets: u32;

@group(0) @binding(24) var<storage, read_write> subtrigs: array<SubTrig>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    
  let offset_3d = mesh.offset_3d_data;
  let n_tets = bitcast<u32>(mesh.data[offset_3d + 0]) + bitcast<u32>(mesh.data[offset_3d + 2]) + 2 * bitcast<u32>(mesh.data[offset_3d + 3]) + 5*bitcast<u32>(mesh.data[offset_3d + 4]);
  
  for (var i = id.x; i<n_tets; i+=256*1024) {
#ifdef REGION_VISIBILITY
    if (regionAlphaVol(getTetrahedron(i).index) == 0.0) {
      continue;
    }
#endif REGION_VISIBILITY
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
    let max_trigs = arrayLength(&subtrigs);
    if(index >= max_trigs) {
      continue;
    }
    let write_count = min(cuts.n, max_trigs - index);
    for(var k = 0u; k < write_count; k++) {
      subtrigs[index+k] = cuts.trigs[k];
    }
  }
}

