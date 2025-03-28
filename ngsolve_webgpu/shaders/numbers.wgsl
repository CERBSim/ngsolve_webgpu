#import font
#import clipping
#import camera

@group(0) @binding(12) var<storage> vertices : array<f32>;

@vertex
fn vertexPointNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) pointId: u32) -> FontFragmentInput {
    var p = vec3<f32>(vertices[3 * pointId], vertices[3 * pointId + 1], vertices[3 * pointId + 2]);

    if calcClipping(p) == false {
        return FontFragmentInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
    }

    var position = cameraMapPoint(p);

    let i_digit = vertexId / 6;
    let vi = vertexId % 6;

    var length = 1u;
    var n = 10u;
    while n <= pointId + 1 {
        length++;
        n *= 10u;
    }

    if i_digit >= length {
        return FontFragmentInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
    }

    var digit = pointId + 1;
    for (var i = 0u; i < i_digit; i++) {
        digit = digit / 10;
    }
    digit = digit % 10;

    position.x += f32(length - i_digit - 1) * u_font.width_normalized * position.w;
    return fontCalc(digit+48, position, vi);
}

