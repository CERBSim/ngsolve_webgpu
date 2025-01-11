
@vertex
fn vertexPointNumber(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) pointId: u32) -> FragmentTextInput {
    var p = vec3<f32>(vertices[3 * pointId], vertices[3 * pointId + 1], vertices[3 * pointId + 2]);

    if calcClipping(p) == false {
        return FragmentTextInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
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
        return FragmentTextInput(vec4<f32>(-1.0, -1.0, 0.0, 1.0), vec2<f32>(0.));
    }

    var digit = pointId + 1;
    for (var i = 0u; i < i_digit; i++) {
        digit = digit / 10;
    }
    digit = digit % 10;

    let w: f32 = 2 * f32(u_font.width) / 400.;
    let h: f32 = 2 * f32(u_font.height) / 400.;

    var tex_coord = vec2<f32>(
        f32((digit + 16) * u_font.width),
        f32(u_font.height)
    );

    if vi == 2 || vi == 4 || vi == 5 {
        position.y += h * position.w;
        tex_coord.y = 0.0;
    }

    position.x += f32(length - i_digit - 1) * w * position.w;

    if vi == 1 || vi == 2 || vi == 4 {
        position.x += w * position.w;
        tex_coord.x += f32(u_font.width);
    }

    return FragmentTextInput(position, tex_coord);
}

@fragment
fn fragmentText(@location(0) tex_coord: vec2<f32>) -> @location(0) vec4<f32> {
    let alpha: f32 = textureLoad(
        u_font_texture,
        vec2i(floor(tex_coord)),
        0
    ).x;

    if alpha < 0.01 {
      discard;
    }

    return vec4(0., 0., 0., alpha);
}

struct FragmentTextInput {
    @builtin(position) fragPosition: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};
