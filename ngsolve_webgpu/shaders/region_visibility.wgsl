// Per-region draw alpha and per-element visibility bits, see
// region_visibility.py for the buffer layout:
//   [n_vol, n_surf, n_el2d, n_el3d (bitcast), vol_alpha[n_vol],
//    surf_alpha[n_surf], el2d_bits[(n_el2d + 31) / 32], el3d_bits[...]]
// An alpha of 0 means "do not draw"; out-of-range indices are fully visible.
// The binding is only part of a pipeline's layout when the renderer has a
// RegionVisibility assigned (its call sites are gated by the REGION_VISIBILITY
// define), so shaders importing this file don't require the buffer otherwise.

@group(0) @binding(34) var<storage> u_region_alpha : array<f32>;

// Alpha for a 3D element region (netgen material number - 1).
fn regionAlphaVol(index: u32) -> f32 {
    let n_vol = bitcast<u32>(u_region_alpha[0]);
    if (index >= n_vol) { return 1.0; }
    return u_region_alpha[4u + index];
}

// Alpha for a 2D element index (face descriptor number - 1 on 3D meshes,
// material number - 1 on 2D meshes).
fn regionAlphaSurf(index: u32) -> f32 {
    let n_vol = bitcast<u32>(u_region_alpha[0]);
    let n_surf = bitcast<u32>(u_region_alpha[1]);
    if (index >= n_surf) { return 1.0; }
    return u_region_alpha[4u + n_vol + index];
}

fn _elementBit(offset: u32, n: u32, nr: u32) -> bool {
    if (nr >= n) { return true; }
    let word = bitcast<u32>(u_region_alpha[offset + nr / 32u]);
    return ((word >> (nr % 32u)) & 1u) == 1u;
}

// Whether the drawn 2D element nr is visible (no mask: all are).
fn elementVisible2d(nr: u32) -> bool {
    let n_vol = bitcast<u32>(u_region_alpha[0]);
    let n_surf = bitcast<u32>(u_region_alpha[1]);
    return _elementBit(4u + n_vol + n_surf, bitcast<u32>(u_region_alpha[2]), nr);
}

// Whether the drawn 3D element nr is visible.
fn elementVisible3d(nr: u32) -> bool {
    let n_vol = bitcast<u32>(u_region_alpha[0]);
    let n_surf = bitcast<u32>(u_region_alpha[1]);
    let n_el2d = bitcast<u32>(u_region_alpha[2]);
    return _elementBit(4u + n_vol + n_surf + (n_el2d + 31u) / 32u,
                       bitcast<u32>(u_region_alpha[3]), nr);
}
