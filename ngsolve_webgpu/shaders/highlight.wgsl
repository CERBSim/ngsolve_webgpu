#ifdef RENDER_OBJECT_ID
struct HighlightUniforms {
    renderer_id: u32,
    element_id: u32,
    region_index: u32,
    solid_index: u32,
};

@group(0) @binding(57) var<uniform> u_highlight: HighlightUniforms;

#ifdef HAS_SELECTION
@group(0) @binding(58) var<storage> u_selection: array<u32>;
#endif HAS_SELECTION

fn highlightColor(color: vec4<f32>, mode: u32) -> vec4<f32> {
    if (mode == 2u) {
        // selected + hover: intense red-orange
        return mix(color, vec4<f32>(1.0, 0.2, 0.0, color.a), 0.65);
    }
    if (mode == 1u) {
        // selected: medium orange
        return mix(color, vec4<f32>(1.0, 0.5, 0.0, color.a), 0.5);
    }
    // hover: light yellow tint
    return mix(color, vec4<f32>(1.0, 0.9, 0.5, color.a), 0.25);
}

fn applyHighlight(color: vec4<f32>, instance_id: u32, region_id: u32) -> vec4<f32> {
    // Selection buffer check first (lower priority, drawn underneath)
    var is_selected = false;
#ifdef HAS_SELECTION
    if (region_id < arrayLength(&u_selection) && u_selection[region_id] != 0u) {
        is_selected = true;
    }
#endif HAS_SELECTION
    // Hover uniform check (higher priority)
    if (u_highlight.renderer_id == @RENDER_OBJECT_ID@ && u_highlight.element_id == instance_id) {
        if (is_selected) { return highlightColor(color, 2u); }
        return highlightColor(color, 0u);
    }
    if (u_highlight.renderer_id == @RENDER_OBJECT_ID@ && u_highlight.region_index == region_id) {
        if (is_selected) { return highlightColor(color, 2u); }
        return highlightColor(color, 0u);
    }
    if (is_selected) { return highlightColor(color, 1u); }
    return color;
}
#else RENDER_OBJECT_ID
fn applyHighlight(color: vec4<f32>, instance_id: u32, region_id: u32) -> vec4<f32> {
    return color;
}
#endif RENDER_OBJECT_ID
