#import camera
#import clipping
#import colormap
#import ngsolve/eval/seg

@group(0) @binding(80) var<storage> u_facet_geometry: array<f32>;
@group(0) @binding(81) var<storage> u_facet_values: array<f32>;
@group(0) @binding(82) var<uniform> u_facet_subdivision: u32;
@group(0) @binding(83) var<uniform> u_facet_thickness: f32;
@group(0) @binding(84) var<uniform> u_facet_deformation_scale: f32;

struct FacetVertexOutput {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) value: f32,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertId: u32,
               @builtin(instance_index) edgeId: u32) -> FacetVertexOutput {
    let subdivision = u_facet_subdivision;
    let order = u32(u_facet_geometry[1]);
    let offset_geometry = u32(u_facet_geometry[2]);
    let offset_opposite = u32(u_facet_geometry[3]);
    let offset_deformation = u32(u_facet_geometry[4]);

    // 2 * (subdivision+1) vertices per instance, forming a triangle strip
    // Even vertices: offset in one direction, odd: offset in other direction
    let seg_idx = vertId / 2u;
    let side = vertId % 2u;
    let lam = f32(seg_idx) / f32(subdivision);

    // Evaluate position via de Casteljau on geometry Bernstein
    var position = evalSegVec3(&u_facet_geometry, edgeId, lam, offset_geometry);

    // Apply deformation if present
    if (offset_deformation != 0u) {
        let deform = evalSegVec3(&u_facet_geometry, edgeId, lam, offset_deformation);
        position += u_facet_deformation_scale * deform;
    }

    // Evaluate function value
    let value = evalSeg(&u_facet_values, edgeId, 0u, lam);

    // Get opposite vertex for offset direction
    let opp = vec3<f32>(
        u_facet_geometry[offset_opposite + edgeId * 3u],
        u_facet_geometry[offset_opposite + edgeId * 3u + 1u],
        u_facet_geometry[offset_opposite + edgeId * 3u + 2u]
    );

    // Compute screen-space offset toward opposite vertex
    let proj_pos = cameraMapPoint(position);
    let proj_opp = cameraMapPoint(opp);

    let screen_pos = proj_pos.xy / proj_pos.w;
    let screen_opp = proj_opp.xy / proj_opp.w;

    // Direction from edge toward opposite vertex (inward)
    var inward = normalize(screen_opp - screen_pos);

    // Offset: side 0 = full inward offset, side 1 = no offset (at edge position)
    let thickness = u_facet_thickness;
    var offset: vec2<f32>;
    if (side == 0u) {
        offset = inward * thickness;
    } else {
        offset = vec2<f32>(0.0, 0.0);
    }

    // Correct for aspect ratio
    offset.x /= u_camera.aspect;

    let final_pos = vec4<f32>(proj_pos.xy + offset * proj_pos.w, proj_pos.zw);

    return FacetVertexOutput(final_pos, position, value);
}

@fragment
fn fragment_main(input: FacetVertexOutput) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let color = getColor(input.value);
    if (color.a < 0.01) {
        discard;
    }
    return color;
}
