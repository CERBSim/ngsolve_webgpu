#import camera
#import light
#import colormap
#import clipping
#import ngsolve/highlight
#import ngsolve/clipping/common
#import ngsolve/eval/hex
#import ngsolve/eval/prism
#ifdef SYMMETRY
#import ngsolve/symmetry
#endif SYMMETRY
#ifdef LIC
#import ngsolve/lic/render
#endif LIC

@group(0) @binding(24) var<storage> subtrigs: array<SubTrig>;
@group(0) @binding(55) var<uniform> u_component: i32;

const CLIP_SUBDIV: u32 = @CLIPPING_SUBDIVISION@;

struct VertexOutputClip {
  @invariant @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) lam: vec3<f32>,
  @location(3) @interpolate(flat) elnr: u32,
  @location(4) @interpolate(flat) value_sign: f32,
  @location(5) @interpolate(flat) index: u32,
};

@vertex
fn vertex_clipping(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) trigId_: u32)
  -> VertexOutputClip
{
#ifdef SYMMETRY
  let trigId = symGetElementIndex(trigId_);
#else SYMMETRY
  let trigId = trigId_;
#endif SYMMETRY
  let trig = subtrigs[trigId];

  // Subdivision grid within the clipping triangle
  let subTrigId = vertId / 3u;
  let localVert = vertId % 3u;
  let h = 1.0 / f32(CLIP_SUBDIV);
  let ix = subTrigId % CLIP_SUBDIV;
  let iy = subTrigId / CLIP_SUBDIV;
  var u = select(0.0, h, localVert == 0u) + f32(ix) * h;
  var v = select(0.0, h, localVert == 1u) + f32(iy) * h;
  if (ix + iy >= CLIP_SUBDIV) {
      u = 1.0 - u;
      v = 1.0 - v;
  }

  // Interpolate lam within the parent clipping triangle
  let ref_lam = (1.0 - u - v) * trig.lam[0] + u * trig.lam[1] + v * trig.lam[2];
  var lam = vec4<f32>(ref_lam, 1.);
  lam[3] = 1.0 - lam[0] - lam[1] - lam[2];

  var p = vec3<f32>(0.0, 0.0, 0.0);
  let tet = getTetrahedron(trig.id);
  let true_map = trueMapOffset(tet.id);
  let is_curved = true_map >= 0;
  if (is_curved) {
      p = evalTrueMap(tet.np, u32(true_map), subTetRefCoord(tet, lam.xyz));
  }
  if (!is_curved) {
      let points = getTetPoints(trig.id);
      for(var i = 0u; i < 4u; i = i + 1u) {
          p = p + lam[i] * points[i];
      }
  }
  if(u_deformation_values_3d[0] != -1 && is_curved) {
      for(var xi = 0u; xi < 3u; xi++) {
          p[xi] += u_deformation_scale * evalTetComplex(&u_deformation_values_3d, trig.id, i32(xi), lam.xyz);
      }
  }
  var n = u_clipping.plane.xyz;
#ifdef SYMMETRY
  p = symApplyPosition(p, trigId_);
  n = symApplyNormal(n, trigId_);
#endif SYMMETRY
  
#ifdef SYMMETRY
  let value_sign = symGetValueSign(trigId_);
#else SYMMETRY
  let value_sign = 1.0;
#endif SYMMETRY
  let tet_for_index = getTetrahedron(trig.id);
  return VertexOutputClip(cameraMapPoint(p), p, n, lam.xyz, trig.id, value_sign, tet_for_index.index);
}

// Barycentric coordinates in sub-tet ei of the drawn point p. Fragments interpolate lam and
// p linearly, for elements with a true map these differ: Newton steps on X(lam) = p.
fn clipFragmentLam(ei: u32, lam_in: vec3f, p: vec3f) -> vec3f {
#ifdef SYMMETRY
  return lam_in;
#else SYMMETRY
  var lam = lam_in;
  if (u_deformation_values_3d[0] != -1) {
    return lam;
  }
  let tet = getTetrahedron(ei);
  let true_map = trueMapOffset(tet.id);
  if (true_map < 0) {
    return lam;
  }
  let r3 = nodeRef(tet.np, tet.pi[3]);
  let d0 = nodeRef(tet.np, tet.pi[0]) - r3;
  let d1 = nodeRef(tet.np, tet.pi[1]) - r3;
  let d2 = nodeRef(tet.np, tet.pi[2]) - r3;
  for (var it = 0; it < 2; it++) {
    let g = evalTrueMapGrad(tet.np, u32(true_map), subTetRefCoord(tet, lam));
    let J0 = g[1] * d0.x + g[2] * d0.y + g[3] * d0.z;
    let J1 = g[1] * d1.x + g[2] * d1.y + g[3] * d1.z;
    let J2 = g[1] * d2.x + g[2] * d2.y + g[3] * d2.z;
    let c0 = cross(J1, J2);
    let c1 = cross(J2, J0);
    let c2 = cross(J0, J1);
    let det = dot(J0, c0);
    if (det == 0.0) {
      break;
    }
    let res = g[0] - p;
    lam -= vec3f(dot(c0, res), dot(c1, res), dot(c2, res)) / det;
  }
  return lam;
#endif SYMMETRY
}

@fragment
fn fragment_clipping(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  let value_raw = evalTet(&u_function_values_3d, input.elnr, u_component, clipFragmentLam(input.elnr, input.lam, input.p)) * input.value_sign;
  // Guard against NaN: the coefficient function can be singular (0/0) at e.g. flow
  // sources/sinks, and a NaN value would be sampled into the colormap at a NaN
  // texture coordinate -> transparent texel. With the premultiplied "over" blend
  // (light.wgsl + canvas.py) the background then bleeds through as a bright spot.
  // Map NaN to the colormap minimum so the fragment stays opaque. NaN != NaN.
  let value = select(value_raw, u_cmap_uniforms.min, value_raw != value_raw);
  var color = applyHighlight(getColor(value), input.elnr, input.index);
#ifdef LIC
  color = licModulate(color, input.fragPosition.xy);
#endif LIC
  return lightCalcColor(input.p, input.n, color);
}

#ifdef SELECT_PIPELINE
@fragment fn select_clipping(
    input: VertexOutputClip
) -> @location(0) vec4<u32> {
    return vec4<u32>(@RENDER_OBJECT_ID@, bitcast<u32>(input.fragPosition.z), input.elnr, input.index);
}
#endif SELECT_PIPELINE
