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
#import ngsolve/lic/common
@group(0) @binding(42) var u_lic_output: texture_2d<f32>;
#endif LIC

@group(0) @binding(24) var<storage> subtrigs: array<SubTrig>;
@group(0) @binding(55) var<uniform> u_component: i32;

const CLIP_SUBDIV: u32 = @CLIPPING_SUBDIVISION@;

struct VertexOutputClip {
  @builtin(position) fragPosition: vec4<f32>,
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
  var is_curved = false;
  if (mesh.is_curved != 0u) {
      let oc3d = mesh.offset_curvature_3d;
      let n_curved = bitcast<i32>(mesh.data[oc3d + 1u]);
      if (n_curved > 0) {
          let tet = getTetrahedron(trig.id);
          let lookup_val = bitcast<i32>(mesh.data[oc3d + 2u + tet.id]);
          if (lookup_val >= 0) {
              let order = bitcast<i32>(mesh.data[oc3d]);
              let n_total = bitcast<u32>(mesh.data[mesh.offset_3d_data]);
              let coeff_start = oc3d + 2u + n_total + u32(lookup_val);

              // Map virtual-tet barycentric → parent reference coords
              var ref_verts: array<vec3f, 4>;
              if (tet.np == 4u) {
                  for (var i = 0; i < 4; i++) { ref_verts[i] = NODE_REF[tet.pi[i]]; }
              } else if (tet.np == 6u) {
                  for (var i = 0; i < 4; i++) { ref_verts[i] = NODE_REF_PRISM[tet.pi[i]]; }
              } else if (tet.np == 8u) {
                  for (var i = 0; i < 4; i++) { ref_verts[i] = NODE_REF_HEX[tet.pi[i]]; }
              }

              let parent_lam = lam.x * ref_verts[0] + lam.y * ref_verts[1]
                             + lam.z * ref_verts[2] + (1.0 - lam.x - lam.y - lam.z) * ref_verts[3];

              if (tet.np == 4u) {
                  p = evalTetVec3AtDirect(coeff_start, order, parent_lam);
              } else if (tet.np == 6u) {
                  p = evalPrismVec3At(coeff_start, order, parent_lam);
              } else if (tet.np == 8u) {
                  p = evalHexVec3At(coeff_start, order, parent_lam);
              }
              is_curved = true;
          }
      }
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

@fragment
fn fragment_clipping(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  let value = evalTet(&u_function_values_3d, input.elnr, u_component, input.lam) * input.value_sign;
  var color = applyHighlight(getColor(value), input.elnr, input.index);
#ifdef LIC
  // Modulate the colormapped value with the precomputed LIC grayscale. The LIC
  // texture is computed in SCREEN space at canvas resolution, so this fragment
  // samples it directly at its own framebuffer pixel (@builtin(position)). lic.y
  // is the coverage mask (0 at rasterisation gaps), so gaps fall back to the
  // flat colour.
  let texel = vec2<i32>(clamp(input.fragPosition.xy,
                              vec2f(0.0),
                              vec2f(f32(u_lic.width) - 1.0, f32(u_lic.height) - 1.0)));
  let lic = textureLoad(u_lic_output, texel, 0);
  let shade = mix(1.0, lic.x, u_lic.contrast * lic.y);
  color = vec4f(color.rgb * shade, color.a);
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
