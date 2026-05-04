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
    if (mode == 1u) {
        // select: darker orange
        return mix(color, vec4<f32>(1.0, 0.55, 0.0, color.a), 0.45);
    }
    // hover: light orange
    return mix(color, vec4<f32>(1.0, 0.8, 0.3, color.a), 0.35);
}

fn applyHighlight(color: vec4<f32>, instance_id: u32, region_id: u32) -> vec4<f32> {
    // Selection buffer check first (lower priority, drawn underneath)
#ifdef HAS_SELECTION
    var base = color;
    if (region_id < arrayLength(&u_selection) && u_selection[region_id] != 0u) {
        base = highlightColor(color, 1u);
    }
#else HAS_SELECTION
    let base = color;
#endif HAS_SELECTION
    // Hover uniform check (higher priority, overrides selection)
    if (u_highlight.renderer_id == @RENDER_OBJECT_ID@ && u_highlight.element_id == instance_id) {
        return highlightColor(color, 0u);
    }
    if (u_highlight.renderer_id == @RENDER_OBJECT_ID@ && u_highlight.region_index == region_id) {
        return highlightColor(color, 0u);
    }
    return base;
}
#else RENDER_OBJECT_ID
fn applyHighlight(color: vec4<f32>, instance_id: u32, region_id: u32) -> vec4<f32> {
    return color;
}
#endif RENDER_OBJECT_ID
