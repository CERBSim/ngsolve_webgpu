
@group(0) @binding(82) var<storage, read_write> subtrigs: array<SubTrig>;


@compute @workgroup_size(1)
fn create_iso_triangles(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = get_tet_points(id.x);
  let f = array(p[0].x-0.5, p[1].x-0.5, p[2].x-0.5, p[3].x-0.5);

  let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));
  // let f = array(function_values[4*id.x],
//                 function_values[4*id.x+1],
  //               function_values[4*id.x+2],
    //             function_values[4*id.x+3]);
  let cuts = clipTet(lam, f, id.x);
  if(cuts.n == 0) {
    return;
  }
  let index = atomicAdd(&counter_isosurface, cuts.n);
  if(only_count == u32(1)) {
    return;
  }
  for(var i = 0u; i < cuts.n; i++) {
    subtrigs[index+i] = cuts.trigs[i];
  }
}

