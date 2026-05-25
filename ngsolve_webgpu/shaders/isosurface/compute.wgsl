#import ngsolve/clipping/common

@group(0) @binding(21) var<storage, read_write> count_trigs: atomic<u32>;
@group(0) @binding(22) var<uniform> u_ntets: u32;
@group(0) @binding(24) var<storage, read_write> subtrigs: array<SubTrig>;
@group(0) @binding(26) var<storage> levelset_values: array<f32>;
@group(0) @binding(27) var<uniform> u_iso_subdivision: u32;

fn my_pow(x: u32, y: u32) -> u32 {
  var res = 1u;
  for(var i = 0u; i < y; i++) {
    res *= x;
  }
  return res;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let n_tets = u_ntets;
  for (var i = id.x; i<n_tets*my_pow(8u,u_iso_subdivision); i+=256u*1024u) {
    var b0 = vec4<f32> (1.0, 0.0, 0.0, 0.0);
    var b1 = vec4<f32> (0.0, 1.0, 0.0, 0.0);
    var b2 = vec4<f32> (0.0, 0.0, 1.0, 0.0);
    var b3 = vec4<f32> (0.0, 0.0, 0.0, 1.0);
    var elnr = i / my_pow(8u,u_iso_subdivision);
    var ind = i;
    for(var level = 0u; level < u_iso_subdivision; level++)
      {
        var sub = ind % 8u;
        ind /= 8u;

        // Compute midpoints of all 6 edges
        let m01 = (b0 + b1) * 0.5;
        let m02 = (b0 + b2) * 0.5;
        let m03 = (b0 + b3) * 0.5;
        let m12 = (b1 + b2) * 0.5;
        let m13 = (b1 + b3) * 0.5;
        let m23 = (b2 + b3) * 0.5;

        // Save original vertices for corner tets
        let v0 = b0; let v1 = b1; let v2 = b2; let v3 = b3;

        switch (sub) {
          // Corner tets: one at each original vertex
          case 0u { b0 = v0;  b1 = m01; b2 = m02; b3 = m03; }
          case 1u { b0 = v1;  b1 = m01; b2 = m12; b3 = m13; }
          case 2u { b0 = v2;  b1 = m02; b2 = m12; b3 = m23; }
          case 3u { b0 = v3;  b1 = m03; b2 = m13; b3 = m23; }
          // Octahedron tets: fan around diagonal m01-m23
          case 4u { b0 = m01; b1 = m23; b2 = m02; b3 = m03; }
          case 5u { b0 = m01; b1 = m23; b2 = m03; b3 = m13; }
          case 6u { b0 = m01; b1 = m23; b2 = m13; b3 = m12; }
          case 7u { b0 = m01; b1 = m23; b2 = m12; b3 = m02; }
          default {}
        }
      }
    let element = getElem(elnr);
    let p = array(getElementVertex(element, 0), getElementVertex(element, 1), getElementVertex(element, 2), getElementVertex(element, 3));
    let lam = array(b0.xyz,
                    b1.xyz,
                    b2.xyz,
                    b3.xyz);
    let f = array(evalTet(&levelset_values, elnr, 0, b0.xyz),
                  evalTet(&levelset_values, elnr, 0, b1.xyz),
                  evalTet(&levelset_values, elnr, 0, b2.xyz),
                  evalTet(&levelset_values, elnr, 0, b3.xyz));
    let cuts = clipTet(lam, f, elnr);
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
