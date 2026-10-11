#import clipping
#import ngsolve/clipping/common
#import ngsolve/eval/tet
#import ngsolve/region_visibility

@group(0) @binding(21) var<storage, read_write> count_trigs: atomic<u32>;
@group(0) @binding(22) var<uniform> u_ntets: u32;

@group(0) @binding(24) var<storage, read_write> subtrigs: array<SubTrig>;

// refinement levels of sub-tets with a true map, the cut is computed on 8^levels children
const CUT_LEVELS: u32 = @CLIPPING_CUT_LEVELS@;

fn writeCuts(cuts: ClipTetResult) {
  if(cuts.n == 0) {
    return;
  }
  let index = atomicAdd(&count_trigs, cuts.n);
  let max_trigs = arrayLength(&subtrigs);
  if(index >= max_trigs) {
    return;
  }
  let write_count = min(cuts.n, max_trigs - index);
  for(var k = 0u; k < write_count; k++) {
    subtrigs[index+k] = cuts.trigs[k];
  }
}

// Barycentric corners of child `child` of the 8^levels red refinement of the unit tet
fn childTet(child: u32, levels: u32) -> array<vec4f, 4> {
  var b = array(vec4f(1.0, 0.0, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0),
                vec4f(0.0, 0.0, 1.0, 0.0), vec4f(0.0, 0.0, 0.0, 1.0));
  var ind = child;
  for (var level = 0u; level < levels; level++) {
    let sub = ind % 8u;
    ind /= 8u;
    let m01 = (b[0] + b[1]) * 0.5;
    let m02 = (b[0] + b[2]) * 0.5;
    let m03 = (b[0] + b[3]) * 0.5;
    let m12 = (b[1] + b[2]) * 0.5;
    let m13 = (b[1] + b[3]) * 0.5;
    let m23 = (b[2] + b[3]) * 0.5;
    switch (sub) {
      case 0u { b = array(b[0], m01, m02, m03); }
      case 1u { b = array(b[1], m01, m12, m13); }
      case 2u { b = array(b[2], m02, m12, m23); }
      case 3u { b = array(b[3], m03, m13, m23); }
      case 4u { b = array(m01, m23, m02, m03); }
      case 5u { b = array(m01, m23, m03, m13); }
      case 6u { b = array(m01, m23, m13, m12); }
      default { b = array(m01, m23, m12, m02); }
    }
  }
  return b;
}

// Move a cut point of clipTet from the linear interpolant onto the true zero of n.X(lam)+d
// along its child edge (Illinois steps), so that cut points on shared faces agree.
fn refineCutPoint(ei: u32, tet: Tetrahedron, true_map: u32, lam: array<vec3f, 4>,
                  f: array<f32, 4>, v: vec3f) -> vec3f {
  var a = 0; var b = 1;
  var best = 1e30;
  for (var i = 0; i < 4; i++) {
    for (var j = i + 1; j < 4; j++) {
      if ((f[i] > 0.0) == (f[j] > 0.0)) {
        continue;
      }
      let e = lam[j] - lam[i];
      let d = v - lam[i];
      let dist = length(d - dot(d, e) / dot(e, e) * e);
      if (dist < best) {
        best = dist; a = i; b = j;
      }
    }
  }
  // search in a canonical direction, so that children sharing the edge get the same point
  let la = lam[a]; let lb = lam[b];
  if (la.x > lb.x || (la.x == lb.x && (la.y > lb.y || (la.y == lb.y && la.z > lb.z)))) {
    let h = a; a = b; b = h;
  }
  var t0 = 0.0; var f0 = f[a];
  var t1 = 1.0; var f1 = f[b];
  var t = f0 / (f0 - f1);
  var side = 0;
  for (var it = 0; it < 6; it++) {
    let ft = dot(vec4<f32>(subTetPoint(ei, tet, true_map, mix(lam[a], lam[b], t)), 1.0), u_clipping.plane);
    if (ft == 0.0) {
      break;
    }
    if ((ft > 0.0) == (f0 > 0.0)) {
      t0 = t; f0 = ft;
      if (side == -1) { f1 *= 0.5; }
      side = -1;
    } else {
      t1 = t; f1 = ft;
      if (side == 1) { f0 *= 0.5; }
      side = 1;
    }
    t = (f0 * t1 - f1 * t0) / (f0 - f1);
  }
  return mix(lam[a], lam[b], t);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

  let offset_3d = mesh.offset_3d_data;
  let n_tets = bitcast<u32>(mesh.data[offset_3d + 0]) + bitcast<u32>(mesh.data[offset_3d + 2]) + 2 * bitcast<u32>(mesh.data[offset_3d + 3]) + 5*bitcast<u32>(mesh.data[offset_3d + 4]);

  for (var i = id.x; i<n_tets; i+=256*1024) {
    let tet = getTetrahedron(i);
#ifdef REGION_VISIBILITY
    if (regionAlphaVol(tet.index) == 0.0) {
      continue;
    }
#endif REGION_VISIBILITY
    let true_map = trueMapOffset(tet.id);
    if (true_map < 0) {
      let p = getTetPoints(i);
      let lam = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));

      let f = array(dot(vec4<f32>(p[0], 1.0), u_clipping.plane),
                    dot(vec4<f32>(p[1], 1.0), u_clipping.plane),
                    dot(vec4<f32>(p[2], 1.0), u_clipping.plane),
                    dot(vec4<f32>(p[3], 1.0), u_clipping.plane));
      writeCuts(clipTet(lam, f, i));
      continue;
    }

    // cut the level set n.X(lam)+d of the true map on the refined children
    var n_children = 1u;
    for (var level = 0u; level < CUT_LEVELS; level++) {
      n_children *= 8u;
    }
    for (var child = 0u; child < n_children; child++) {
      let b = childTet(child, CUT_LEVELS);
      let lam = array(b[0].xyz, b[1].xyz, b[2].xyz, b[3].xyz);
      var f: array<f32, 4>;
      for (var k = 0; k < 4; k++) {
        f[k] = dot(vec4<f32>(subTetPoint(i, tet, u32(true_map), lam[k]), 1.0), u_clipping.plane);
      }
      var cuts = clipTet(lam, f, i);
      for (var k = 0u; k < cuts.n; k++) {
        for (var m = 0; m < 3; m++) {
          cuts.trigs[k].lam[m] = refineCutPoint(i, tet, u32(true_map), lam, f, cuts.trigs[k].lam[m]);
        }
      }
      writeCuts(cuts);
    }
  }
}
