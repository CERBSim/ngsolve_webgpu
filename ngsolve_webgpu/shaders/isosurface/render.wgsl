#import ngsolve/clipping/render
#ifdef SYMMETRY
#import ngsolve/symmetry
#endif SYMMETRY

@group(0) @binding(26) var<storage> levelset_values: array<f32>;

@vertex
fn vertex_isosurface(@builtin(vertex_index) vertId: u32,
                     @builtin(instance_index) trigId_: u32)
  -> VertexOutputClip
{
#ifdef SYMMETRY
  let trigId = symGetElementIndex(trigId_);
#else SYMMETRY
  let trigId = trigId_;
#endif SYMMETRY
  let trig = subtrigs[trigId];

  var lam = vec4<f32>(trig.lam[vertId], 1.0);
  lam[3] = 1.0 - lam[0] - lam[1] - lam[2];

  // Compute position (with curved element support)
  var p = vec3<f32>(0.0, 0.0, 0.0);
  var is_curved = false;

  // Jacobian columns (dp/dlam_i), needed for normal computation
  var J0 = vec3f(0.0);
  var J1 = vec3f(0.0);
  var J2 = vec3f(0.0);

  let tet = getTetrahedron(trig.id);
  let true_map = trueMapOffset(tet.id);
  if (true_map >= 0) {
      let result = evalTrueMapGrad(tet.np, u32(true_map), subTetRefCoord(tet, lam.xyz));
      p = result[0];

      // Jacobian: dp/dlam_i via chain rule through parent reference coords
      let r3 = nodeRef(tet.np, tet.pi[3]);
      let dpl0 = nodeRef(tet.np, tet.pi[0]) - r3;
      let dpl1 = nodeRef(tet.np, tet.pi[1]) - r3;
      let dpl2 = nodeRef(tet.np, tet.pi[2]) - r3;
      J0 = result[1] * dpl0.x + result[2] * dpl0.y + result[3] * dpl0.z;
      J1 = result[1] * dpl1.x + result[2] * dpl1.y + result[3] * dpl1.z;
      J2 = result[1] * dpl2.x + result[2] * dpl2.y + result[3] * dpl2.z;

      is_curved = true;
  }
  if (!is_curved) {
      let points = getTetPoints(trig.id);
      for(var i = 0u; i < 4u; i = i + 1u) {
          p = p + lam[i] * points[i];
      }
      J0 = points[0] - points[3];
      J1 = points[1] - points[3];
      J2 = points[2] - points[3];
  }
  if(u_deformation_values_3d[0] != -1 && is_curved) {
      for(var xi = 0u; xi < 3u; xi++) {
          p[xi] += u_deformation_scale * evalTetComplex(&u_deformation_values_3d, trig.id, i32(xi), lam.xyz);
      }
  }

  // Normal from levelset gradient (smooth per-vertex, consistent orientation)
  let eps = 0.001;
  let grad_ref = vec3f(
      evalTet(&levelset_values, trig.id, 0, lam.xyz + vec3f(eps, 0.0, 0.0))
        - evalTet(&levelset_values, trig.id, 0, lam.xyz - vec3f(eps, 0.0, 0.0)),
      evalTet(&levelset_values, trig.id, 0, lam.xyz + vec3f(0.0, eps, 0.0))
        - evalTet(&levelset_values, trig.id, 0, lam.xyz - vec3f(0.0, eps, 0.0)),
      evalTet(&levelset_values, trig.id, 0, lam.xyz + vec3f(0.0, 0.0, eps))
        - evalTet(&levelset_values, trig.id, 0, lam.xyz - vec3f(0.0, 0.0, eps))
  );
  // Transform to physical space via adjugate of Jacobian transpose
  let n_unnorm = grad_ref.x * cross(J1, J2)
               + grad_ref.y * cross(J2, J0)
               + grad_ref.z * cross(J0, J1);
  var n = normalize(n_unnorm);

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
  return VertexOutputClip(cameraMapPoint(p), p, -n, lam.xyz,
                      trig.id, value_sign, tet_for_index.index);
}


@fragment
fn fragment_isosurface(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  checkClipping(input.p);
  let value = evalTet(&u_function_values_3d, input.elnr, 0, input.lam) * input.value_sign;
  return lightCalcColor(input.p, input.n, getColor(value));
}
