#import font
#import clipping
#import camera
#import ngsolve/mesh/utils

@group(0) @binding(12) var<storage> u_edges : array<u32>;
@group(0) @binding(13) var<storage> u_facets : array<u32>;
@group(0) @binding(14) var<uniform> u_number_offset : u32;

fn numDigits(n: u32) -> u32 {
    var length = 1u;
    var threshold = 10u;
    while threshold <= n {
        length++;
        threshold *= 10u;
    }
    return length;
}

fn extractDigit(n: u32, i_digit: u32) -> u32 {
    var d = n;
    for (var i = 0u; i < i_digit; i++) {
        d = d / 10u;
    }
    return d % 10u;
}

fn renderNumber(p: vec3f, number: u32, vertexId: u32) -> FontFragmentInput {
    if calcClipping(p) == false {
        return FontFragmentInput(vec4f(-1.0, -1.0, 0.0, 1.0), vec2f(0.0));
    }

    var position = cameraMapPoint(p);

    let i_digit = vertexId / 6u;
    let vi = vertexId % 6u;

    let length = numDigits(number);
    if i_digit >= length {
        return FontFragmentInput(vec4f(-1.0, -1.0, 0.0, 1.0), vec2f(0.0));
    }

    let digit = extractDigit(number, i_digit);
    let char_size = fontGetSizeOnScreen();
    position.x += f32(length - i_digit - 1u) * char_size.z * position.w;

    // char_map['0'] = 16. fontCalc(0) is the empty sentinel (space).
    // So fontCalc(16) = '0', fontCalc(17) = '1', ..., fontCalc(25) = '9'.
    return fontCalc(digit + 16u, position, vi);
}

@vertex
fn vertexVertexNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) entityId: u32) -> FontFragmentInput {
    let p = getVertex(entityId);
    return renderNumber(p, entityId + u_number_offset, vertexId);
}

@vertex
fn vertexEdgeNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) entityId: u32) -> FontFragmentInput {
    let v0 = u_edges[2u * entityId];
    let v1 = u_edges[2u * entityId + 1u];
    let p = 0.5 * (getVertex(v0) + getVertex(v1));
    return renderNumber(p, entityId + u_number_offset, vertexId);
}

@vertex
fn vertexFacetNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) entityId: u32) -> FontFragmentInput {
    let nv = u_facets[4u * entityId + 3u];
    let v0 = u_facets[4u * entityId];
    let v1 = u_facets[4u * entityId + 1u];
    let v2 = u_facets[4u * entityId + 2u];
    var p = (getVertex(v0) + getVertex(v1) + getVertex(v2)) / 3.0;
    if nv == 4u {
        // quad facet: average 4 vertices — but we only stored 3 + count
        // For now use 3-vertex centroid (close enough for label placement)
        // TODO: store 4th vertex if needed
    }
    return renderNumber(p, entityId + u_number_offset, vertexId);
}

@vertex
fn vertexSurfaceElementNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) entityId: u32) -> FontFragmentInput {
    let tri = loadTriangle(entityId);
    let p = (tri.p[0] + tri.p[1] + tri.p[2]) / 3.0;
    return renderNumber(p, entityId + u_number_offset, vertexId);
}

@vertex
fn vertexVolumeElementNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) entityId: u32) -> FontFragmentInput {
    let elem = getElem(entityId);
    let p = getCenter(elem);
    return renderNumber(p, entityId + u_number_offset, vertexId);
}
