// Per-region draw alpha, see region_visibility.py for the buffer layout:
//   [bitcast(n_vol), bitcast(n_surf), vol_alpha[n_vol], surf_alpha[n_surf]]
// An alpha of 0 means "do not draw"; out-of-range indices are fully visible.
// The binding is only part of a pipeline's layout when the renderer has a
// RegionVisibility assigned (its call sites are gated by the REGION_VISIBILITY
// define), so shaders importing this file don't require the buffer otherwise.

@group(0) @binding(34) var<storage> u_region_alpha : array<f32>;

// Alpha for a 3D element region (netgen material number - 1).
fn regionAlphaVol(index: u32) -> f32 {
    let n_vol = bitcast<u32>(u_region_alpha[0]);
    if (index >= n_vol) { return 1.0; }
    return u_region_alpha[2u + index];
}

// Alpha for a 2D element index (face descriptor number - 1 on 3D meshes,
// material number - 1 on 2D meshes).
fn regionAlphaSurf(index: u32) -> f32 {
    let n_vol = bitcast<u32>(u_region_alpha[0]);
    let n_surf = bitcast<u32>(u_region_alpha[1]);
    if (index >= n_surf) { return 1.0; }
    return u_region_alpha[2u + n_vol + index];
}
