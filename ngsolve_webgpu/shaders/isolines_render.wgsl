#import ngsolve/mesh/render
#import ngsolve/isolines

@fragment
fn fragmentIsolines(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &u_function_values_2d;
    let value = evalTrig(p, input.instanceId, u_function_component, input.lam) * input.value_sign;

    if (u_isolines.n_lines == 0u) {
        // No isolines — just render the field (fallback)
        if (u_isolines.show_field == 0u) { discard; }
        let color = getColor(value);
        if (color.a < 0.01) { discard; }
        return lightCalcColor(input.p, input.n, applyHighlight(color, input.instanceId, input.index));
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
        // Colored field with isolines blended on top
        var color = getColor(value);
        if (color.a < 0.01) { discard; }
        color = mix(u_isolines.color, color, t);
        return lightCalcColor(input.p, input.n, applyHighlight(color, input.instanceId, input.index));
    } else {
        // Isolines only — discard non-isoline pixels
        if (t > 0.95) { discard; }
        let alpha = (1.0 - t) * u_isolines.color.a;
        return vec4f(u_isolines.color.rgb * alpha, alpha);
    }
}
