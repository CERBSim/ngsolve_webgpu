#import ngsolve/clipping/common

@group(0) @binding(21) var<storage, read_write> count_trigs: atomic<u32>;
@group(0) @binding(22) var<uniform> u_ntets: u32;
@group(0) @binding(23) var<uniform> only_count: u32;
@group(0) @binding(24) var<storage, read_write> subtrigs: array<SubTrig>;
@group(0) @binding(26) var<storage> levelset_values: array<f32>;
@group(0) @binding(27) var<uniform> u_subdivision: u32;

fn my_pow(x: u32, y: u32) -> u32 {
  var res = 1u;
  for(var i = 0u; i < y; i++) {
    res *= x;
  }
  return res;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  for (var i = id.x; i<u_ntets*my_pow(4u,u_subdivision); i+=256u*1024u) {
    var b0 = vec4<f32> (1.0, 0.0, 0.0, 0.0);
    var b1 = vec4<f32> (0.0, 1.0, 0.0, 0.0);
    var b2 = vec4<f32> (0.0, 0.0, 1.0, 0.0);
    var b3 = vec4<f32> (0.0, 0.0, 0.0, 1.0);
    var elnr = i / my_pow(4u,u_subdivision);
    var scale: f32 = 1.;
    var ind = i;
    for(var level = 0u; level < u_subdivision; level++)
      {
        var b = ind % 4u;
        ind /= 4u;
        if(b == 0u) {
          b0 = (b0 + b1) * 0.5;
        } else if(b == 1u) {
          b1 = (b0 + b1) * 0.5;
        } else if(b == 2u) {
          b2 = (b2 + b3) * 0.5;
        } else if(b == 3u) {
          b3 = (b2 + b3) * 0.5;
        }
      }
    let p = get_tet_points(elnr);
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
    if(only_count == u32(1)) {
      continue;
    }
    for(var k = 0u; k < cuts.n; k++) {
      subtrigs[index+k] = cuts.trigs[k];
    }
  }
}

