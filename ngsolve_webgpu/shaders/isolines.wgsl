struct IsolineUniforms {
    n_lines: u32,
    thickness: f32,
    show_field: u32,
    padding: u32,
    color: vec4f,
};

@group(0) @binding(60) var<uniform> u_isolines: IsolineUniforms;
