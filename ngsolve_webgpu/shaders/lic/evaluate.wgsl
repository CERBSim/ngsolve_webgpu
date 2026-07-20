// LIC stage 1: evaluate the clipping-plane vector field into a SCREEN-SPACE
// texture.
//
// This is the compute-shader equivalent of "rendering the clipping plane into a
// texture": instead of an offscreen render pass (not available in the export
// engine) we iterate the volume elements, clip each against the plane (reusing
// the exact same clipTet used by ClippingCF) and software-rasterise every
// resulting cut-triangle into the field texture. Unlike the original
// plane-parameter version, the rasterisation happens in framebuffer-pixel space
// under the live camera (perspective projection), so the field texture is the
// size of the canvas and must be recomputed whenever the camera moves. For each
// covered pixel we store the SCREEN-SPACE flow direction and a coverage mask:
//
//     field.xy = unit screen-space flow direction (camera Jacobian of in-plane flow)
//     field.z  = world-space in-plane flow speed |flow|, used for the slow-flow fade
//     field.w  = coverage mask (1 = inside the cut plane, 0 = outside)
//
// The import of ngsolve/clipping/common transitively provides getTetPoints,
// clipTet and evalTet together with the mesh / function / deformation bindings;
// #import camera provides cameraMapPoint for the world->screen projection.

#import clipping
#import ngsolve/clipping/common
#import ngsolve/lic/common
#import ngsolve/eval/trig
#import ngsolve/mesh/utils
#import ngsolve/region_visibility
#import camera

@group(0) @binding(41) var u_lic_field: texture_storage_2d<rgba32float, write>;

// Project a world point to framebuffer-pixel coordinates that match the fragment
// shader's @builtin(position) (origin top-left, y downward). The returned z
// carries clip-space w: z <= 0 means the point is behind the camera and the xy
// pixel coordinates are meaningless.
fn worldToPixel(p: vec3f) -> vec3f {
    let clip = cameraMapPoint(p);
    if (clip.w <= 0.0) {
        return vec3f(0.0, 0.0, clip.w);
    }
    let ndc = clip.xyz / clip.w;
    return vec3f((ndc.x * 0.5 + 0.5) * f32(u_lic.width),
                 (0.5 - 0.5 * ndc.y) * f32(u_lic.height),
                 clip.w);
}

// Number of "virtual" tetrahedra spanning every volume element type, matching
// the iteration space of ngsolve/clipping/compute.wgsl.
fn licNumTets() -> u32 {
    let o = mesh.offset_3d_data;
    return bitcast<u32>(mesh.data[o + 0u])
         + bitcast<u32>(mesh.data[o + 2u])
         + 2u * bitcast<u32>(mesh.data[o + 3u])
         + 5u * bitcast<u32>(mesh.data[o + 4u]);
}

// Clear pass: reset the whole field texture to "no coverage". Run before the
// scatter pass since the scatter only touches covered texels.
@compute @workgroup_size(16, 16, 1)
fn clear_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u_lic.width || gid.y >= u_lic.height) {
        return;
    }
    textureStore(u_lic_field, vec2<i32>(gid.xy), vec4f(0.0, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(256)
fn evaluate_lic_field(@builtin(global_invocation_id) id: vec3<u32>) {
    let n_tets = licNumTets();
    let W = i32(u_lic.width);
    let H = i32(u_lic.height);
    let ncomp = u32(u_function_values_3d[0]);
    let t1 = u_lic.tangent1.xyz;
    let t2 = u_lic.tangent2.xyz;
    // World-space step for the screen-space Jacobian finite difference of the
    // flow direction (~2% of the bounding-sphere radius; inv_extent = 1/(2 R)).
    let eps = 0.01 / u_lic.inv_extent;

    for (var tetId = id.x; tetId < n_tets; tetId += 256u * 1024u) {
#ifdef REGION_VISIBILITY
        if (regionAlphaVol(getTetrahedron(tetId).index) == 0.0) {
            continue;
        }
#endif REGION_VISIBILITY
        let p_tet = getTetPoints(tetId);
        let lam_node = array(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0),
                             vec3f(0.0, 0.0, 1.0), vec3f(0.0, 0.0, 0.0));
        let f = array(dot(vec4<f32>(p_tet[0], 1.0), u_clipping.plane),
                      dot(vec4<f32>(p_tet[1], 1.0), u_clipping.plane),
                      dot(vec4<f32>(p_tet[2], 1.0), u_clipping.plane),
                      dot(vec4<f32>(p_tet[3], 1.0), u_clipping.plane));
        let cuts = clipTet(lam_node, f, tetId);

        for (var ic = 0u; ic < cuts.n; ic++) {
            let lt = cuts.trigs[ic].lam;  // tet-barycentric coords of the 3 verts

            // World positions of the cut-triangle vertices.
            var wp: array<vec3f, 3>;
            for (var k = 0u; k < 3u; k++) {
                let l = lt[k];
                wp[k] = l.x * p_tet[0] + l.y * p_tet[1] + l.z * p_tet[2]
                      + (1.0 - l.x - l.y - l.z) * p_tet[3];
            }

            // Triangle in framebuffer-pixel space (perspective). Skip the whole
            // triangle if any vertex is behind the camera (w <= 0): the cut plane
            // is normally fully in front, so this only drops degenerate straddling
            // cases rather than anything visible.
            var sp: array<vec2f, 3>;
            var skip = false;
            for (var k = 0u; k < 3u; k++) {
                let pix = worldToPixel(wp[k]);
                if (pix.z <= 0.0) { skip = true; }
                sp[k] = pix.xy;
            }
            if (skip) { continue; }

            let a = sp[0];
            let b = sp[1];
            let c = sp[2];

            let lo = vec2<i32>(floor(min(min(a, b), c)));
            let hi = vec2<i32>(ceil(max(max(a, b), c)));
            let x0 = max(lo.x, 0);
            let y0 = max(lo.y, 0);
            let x1 = min(hi.x, W - 1);
            let y1 = min(hi.y, H - 1);

            let det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
            if (abs(det) < 1e-9) {
                continue;
            }
            let invDet = 1.0 / det;

            // Evaluate the field ONCE at the three cut-triangle vertices and map
            // its in-plane component to a SCREEN-SPACE direction via the camera's
            // local Jacobian (finite difference of worldToPixel along the flow),
            // then barycentric-interpolate per pixel. One thread rasterises a
            // whole triangle, so a per-pixel evalTet (expensive de Casteljau /
            // factorial loops, x4 components) would blow past the GPU watchdog;
            // across one small cut triangle the in-plane flow is smooth, so the
            // linear fit of the per-vertex values is visually identical.
            var vdir: array<vec2f, 3>;
            var vspeed: array<f32, 3>;
            var bad = false;
            for (var k = 0u; k < 3u; k++) {
                let lamTet = lt[k];
                var vv = vec3f(0.0, 0.0, 0.0);
                vv[0] = evalTet(&u_function_values_3d, tetId, 0, lamTet);
                if (ncomp > 1u) {
                    vv[1] = evalTet(&u_function_values_3d, tetId, 1, lamTet);
                }
                if (ncomp > 2u) {
                    vv[2] = evalTet(&u_function_values_3d, tetId, 2, lamTet);
                }
                // No-value / NaN: drop the whole triangle so the LIC kernel stops
                // here (the field stays mask = 0 from clear_field). NaN != NaN.
                if (vv.x != vv.x || vv.y != vv.y || vv.z != vv.z) {
                    bad = true;
                }
                // In-plane world flow: its UNIT screen-space direction (camera
                // Jacobian via finite difference; the convolution re-normalises so
                // only the direction matters) and its world-space speed |flow|.
                // The speed is stored so the convolution can fade the LIC toward
                // the flat colour where the flow is very slow (vortex cores /
                // stagnation) — that is where short, coiled streamlines sample too
                // few distinct noise values and produce the bright speckle.
                let fw = dot(vv, t1) * t1 + dot(vv, t2) * t2;
                let fl = length(fw);
                var d = vec2f(0.0, 0.0);
                if (fl > 1e-12) {
                    let sd = worldToPixel(wp[k] + eps * (fw / fl)).xy - sp[k];
                    let sl = length(sd);
                    if (sl > 1e-9) { d = sd / sl; }
                }
                vdir[k] = d;
                vspeed[k] = fl;
            }
            if (bad) { continue; }

            for (var py = y0; py <= y1; py++) {
                for (var px = x0; px <= x1; px++) {
                    let pt = vec2f(f32(px) + 0.5, f32(py) + 0.5);
                    let w0 = ((b.y - c.y) * (pt.x - c.x) + (c.x - b.x) * (pt.y - c.y)) * invDet;
                    let w1 = ((c.y - a.y) * (pt.x - c.x) + (a.x - c.x) * (pt.y - c.y)) * invDet;
                    let w2 = 1.0 - w0 - w1;
                    if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) {
                        continue;
                    }
                    let dir = w0 * vdir[0] + w1 * vdir[1] + w2 * vdir[2];
                    let speed = w0 * vspeed[0] + w1 * vspeed[1] + w2 * vspeed[2];
                    textureStore(u_lic_field, vec2<i32>(px, py), vec4f(dir, speed, 1.0));
                }
            }
        }
    }
}

// Surface variant (cf.dim == mesh.dim == 2): rasterise the 2D mesh's surface
// triangles directly — there is no cutting plane, so we iterate the surface
// elements (no clipTet) and software-rasterise each whole triangle into the same
// screen-space field texture. The in-plane flow IS the surface vector field, so
// no plane projection is needed: the world flow is projected straight to screen
// via the camera Jacobian (exactly as in evaluate_lic_field after the cut). The
// field texture stores the same (screen_dir.xy, speed, mask) layout the
// convolution pass consumes, so line_integral_convolution.wgsl is unchanged.
@compute @workgroup_size(256)
fn evaluate_lic_field_surface(@builtin(global_invocation_id) id: vec3<u32>) {
    let W = i32(u_lic.width);
    let H = i32(u_lic.height);
    // World-space step for the screen-space Jacobian finite difference of the
    // flow direction (~2% of the bounding-sphere radius; inv_extent = 1/(2 R)).
    let eps = 0.01 / u_lic.inv_extent;

    // Total render triangles = number of 2D elements (first triangle of each) +
    // number of quads (their second triangle); this matches the instanceId
    // addressing loadTriangle expects.
    let numElements = bitcast<u32>(mesh.data[mesh.offset_2d_data]);
    let n_trigs = numElements + mesh.num_quads;

    // Reference (eval) coords of the three triangle corners p[0], p[1], p[2]; see
    // the mapping in surface_vectors.wgsl / mesh render's calcTrig.
    let corner_lam = array<vec2f, 3>(vec2f(1.0, 0.0), vec2f(0.0, 1.0), vec2f(0.0, 0.0));

    for (var instanceId = id.x; instanceId < n_trigs; instanceId += 256u * 1024u) {
        let tri = loadTriangle(instanceId);
#ifdef REGION_VISIBILITY
        if (regionAlphaSurf(tri.index) == 0.0) {
            continue;
        }
#endif REGION_VISIBILITY
        let trigId = tri.nr;

        // World positions of the three corners (curvature-aware, like the renderer).
        var wp: array<vec3f, 3>;
        if (mesh.is_curved == 1u) {
            for (var k = 0u; k < 3u; k++) {
                wp[k] = evalTrigVec3(&mesh.data, trigId, corner_lam[k], mesh.offset_curvature_2d);
            }
        } else {
            wp = tri.p;
        }

        // Triangle in framebuffer-pixel space (perspective). Skip if any vertex is
        // behind the camera (w <= 0): its pixel coords are meaningless.
        var sp: array<vec2f, 3>;
        var skip = false;
        for (var k = 0u; k < 3u; k++) {
            let pix = worldToPixel(wp[k]);
            if (pix.z <= 0.0) { skip = true; }
            sp[k] = pix.xy;
        }
        if (skip) { continue; }

        let a = sp[0];
        let b = sp[1];
        let c = sp[2];

        let lo = vec2<i32>(floor(min(min(a, b), c)));
        let hi = vec2<i32>(ceil(max(max(a, b), c)));
        let x0 = max(lo.x, 0);
        let y0 = max(lo.y, 0);
        let x1 = min(hi.x, W - 1);
        let y1 = min(hi.y, H - 1);

        let det = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if (abs(det) < 1e-9) {
            continue;
        }
        let invDet = 1.0 / det;

        // Per-corner screen-space flow direction + world speed. Evaluate the
        // vector field ONCE at each corner (real part) and map its world flow to a
        // screen-space unit direction via the camera Jacobian, then barycentric-
        // interpolate per pixel (see evaluate_lic_field for the rationale).
        var vdir: array<vec2f, 3>;
        var vspeed: array<f32, 3>;
        var bad = false;
        for (var k = 0u; k < 3u; k++) {
            let vv = evalTrigVec3ReIm(&u_function_values_2d, trigId, corner_lam[k], 0u).re;
            // No-value / NaN: drop the whole triangle so the kernel stops here.
            if (vv.x != vv.x || vv.y != vv.y || vv.z != vv.z) {
                bad = true;
            }
            let fl = length(vv);
            var d = vec2f(0.0, 0.0);
            if (fl > 1e-12) {
                let sd = worldToPixel(wp[k] + eps * (vv / fl)).xy - sp[k];
                let sl = length(sd);
                if (sl > 1e-9) { d = sd / sl; }
            }
            vdir[k] = d;
            vspeed[k] = fl;
        }
        if (bad) { continue; }

        for (var py = y0; py <= y1; py++) {
            for (var px = x0; px <= x1; px++) {
                let pt = vec2f(f32(px) + 0.5, f32(py) + 0.5);
                let w0 = ((b.y - c.y) * (pt.x - c.x) + (c.x - b.x) * (pt.y - c.y)) * invDet;
                let w1 = ((c.y - a.y) * (pt.x - c.x) + (a.x - c.x) * (pt.y - c.y)) * invDet;
                let w2 = 1.0 - w0 - w1;
                if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) {
                    continue;
                }
                let dir = w0 * vdir[0] + w1 * vdir[1] + w2 * vdir[2];
                let speed = w0 * vspeed[0] + w1 * vspeed[1] + w2 * vspeed[2];
                textureStore(u_lic_field, vec2<i32>(px, py), vec4f(dir, speed, 1.0));
            }
        }
    }
}
