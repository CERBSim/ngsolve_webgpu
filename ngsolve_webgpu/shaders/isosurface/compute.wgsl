@group(0) @binding(82) var<storage, read_write> subtrigs: array<SubTrig>;

@compute @workgroup_size(256)
fn create_iso_triangles(@builtin(global_invocation_id) id: vec3<u32>) {
  for (var i = id.x; i<u_ntets; i+=256*1024) {
    let p = get_tet_points(i);
    let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));
    let f = array(function_values[4*i],
                  function_values[4*i+1],
                  function_values[4*i+2],
                  function_values[4*i+3]);
    let cuts = clipTet(lam, f, i);
    if(cuts.n == 0) {
      continue;
    }
    let index = atomicAdd(&counter_isosurface, cuts.n);
    if(only_count == u32(1)) {
      continue;
    }
    for(var i = 0u; i < cuts.n; i++) {
      subtrigs[index+i] = cuts.trigs[i];
    }
  }
}

