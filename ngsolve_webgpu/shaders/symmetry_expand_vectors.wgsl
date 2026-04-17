#import ngsolve/symmetry

@group(0) @binding(0) var<storage> src_positions: array<f32>;
@group(0) @binding(1) var<storage> src_directions: array<f32>;
@group(0) @binding(2) var<storage> src_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst_positions: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_directions: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst_values: array<f32>;

@compute @workgroup_size(256)
fn expand_vectors(@builtin(global_invocation_id) id: vec3<u32>) {
    let n_vectors = sym_info.y;
    let n_copies = sym_info.x;
    let total = n_vectors * n_copies;

    for (var i = id.x; i < total; i += 256u * 1024u) {
        let vec_id = i % n_vectors;
        let copy_id = i / n_vectors;
        let s = sym_transforms[copy_id];

        let pos = vec3f(src_positions[vec_id * 3u], src_positions[vec_id * 3u + 1u], src_positions[vec_id * 3u + 2u]);
        let dir = vec3f(src_directions[vec_id * 3u], src_directions[vec_id * 3u + 1u], src_directions[vec_id * 3u + 2u]);

        let new_pos = (s.transform * vec4(pos, 1.0)).xyz;
#ifdef AXIAL_VECTORS
        let new_dir = s.det_sign * (s.transform * vec4(dir, 0.0)).xyz;
#else AXIAL_VECTORS
        let new_dir = (s.transform * vec4(dir, 0.0)).xyz;
#endif AXIAL_VECTORS

        dst_positions[i * 3u]     = new_pos.x;
        dst_positions[i * 3u + 1u] = new_pos.y;
        dst_positions[i * 3u + 2u] = new_pos.z;
        dst_directions[i * 3u]     = new_dir.x;
        dst_directions[i * 3u + 1u] = new_dir.y;
        dst_directions[i * 3u + 2u] = new_dir.z;
        dst_values[i] = src_values[vec_id] * s.value_sign;
    }
}
