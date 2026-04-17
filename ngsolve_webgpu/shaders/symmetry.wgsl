struct SymmetryTransform {
    transform: mat4x4<f32>,
    det_sign: f32,
    value_sign: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(120) var<storage> sym_transforms: array<SymmetryTransform>;
@group(0) @binding(121) var<uniform> sym_info: vec4<u32>; // x=n_copies, y=n_elements

fn symGetElementIndex(instance_index: u32) -> u32 {
    return instance_index % sym_info.y;
}

fn symGetCopyIndex(instance_index: u32) -> u32 {
    return instance_index / sym_info.y;
}

fn symApplyPosition(pos: vec3f, instance_index: u32) -> vec3f {
    let idx = symGetCopyIndex(instance_index);
    return (sym_transforms[idx].transform * vec4(pos, 1.0)).xyz;
}

fn symApplyNormal(n: vec3f, instance_index: u32) -> vec3f {
    let idx = symGetCopyIndex(instance_index);
    // No det_sign: outward normals stay outward after mirroring.
    // M * n gives the correct outward normal for the mirrored surface.
    return (sym_transforms[idx].transform * vec4(n, 0.0)).xyz;
}

fn symApplyDirection(dir: vec3f, instance_index: u32) -> vec3f {
    let idx = symGetCopyIndex(instance_index);
    return (sym_transforms[idx].transform * vec4(dir, 0.0)).xyz;
}

fn symApplyAxialDirection(dir: vec3f, instance_index: u32) -> vec3f {
    let idx = symGetCopyIndex(instance_index);
    let s = sym_transforms[idx];
    // Axial vectors (pseudovectors like B-field): transform = det(M) * M
    return s.det_sign * (s.transform * vec4(dir, 0.0)).xyz;
}

fn symGetValueSign(instance_index: u32) -> f32 {
    let idx = symGetCopyIndex(instance_index);
    return sym_transforms[idx].value_sign;
}
