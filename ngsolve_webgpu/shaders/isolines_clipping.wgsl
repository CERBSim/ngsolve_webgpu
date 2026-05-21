#import ngsolve/clipping/render
#import ngsolve/isolines

@fragment
fn fragmentClippingIsolines(input: VertexOutputClip) -> @location(0) vec4<f32>
{
  let value = evalTet(&u_function_values_3d, input.elnr, u_component, input.lam) * input.value_sign;

  if (u_isolines.n_lines == 0u) {
      if (u_isolines.show_field == 0u) { discard; }
      return lightCalcColor(input.p, input.n, applyHighlight(getColor(value), input.elnr, input.index));
  }

  let range = u_cmap_uniforms.max - u_cmap_uniforms.min;
  if (range <= 0.0) { discard; }

  let spacing = range / f32(u_isolines.n_lines + 1u);
  let shifted = (value - u_cmap_uniforms.min) / spacing;
  let d = abs(fract(shifted + 0.5) - 0.5) * spacing;
  let fw = fwidth(value);
  if (fw <= 0.0) { discard; }

  let t = smoothstep(0.0, u_isolines.thickness * fw, d);

  if (u_isolines.show_field == 1u) {
      var color = getColor(value);
      color = mix(u_isolines.color, color, t);
      return lightCalcColor(input.p, input.n, applyHighlight(color, input.elnr, input.index));
  } else {
      if (t > 0.95) { discard; }
      let alpha = (1.0 - t) * u_isolines.color.a;
      return vec4f(u_isolines.color.rgb * alpha, alpha);
  }
}
