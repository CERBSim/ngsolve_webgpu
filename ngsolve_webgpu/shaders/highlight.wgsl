#ifdef RENDER_OBJECT_ID
struct HighlightUniforms {
    renderer_id: u32,
    element_id: u32,
    region_index: u32,
    _pad: u32,
};

@group(0) @binding(57) var<uniform> u_highlight: HighlightUniforms;

fn applyHighlight(color: vec4<f32>, instance_id: u32, region_id: u32) -> vec4<f32> {
    if (u_highlight.renderer_id == @RENDER_OBJECT_ID@ && u_highlight.element_id == instance_id) {
        return mix(color, vec4<f32>(1.0, 1.0, 1.0, color.a), 0.4);
    }
    if (u_highlight.region_index == region_id) {
        return mix(color, vec4<f32>(1.0, 1.0, 1.0, color.a), 0.4);
    }
    return color;
}
#else RENDER_OBJECT_ID
fn applyHighlight(color: vec4<f32>, instance_id: u32, region_id: u32) -> vec4<f32> {
    return color;
}
#endif RENDER_OBJECT_ID
