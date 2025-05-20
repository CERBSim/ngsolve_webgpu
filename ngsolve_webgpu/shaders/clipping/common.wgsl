#import ngsolve/eval/tet

@group(0) @binding(12) var<storage> vertices : array<f32>;
@group(0) @binding(25) var<storage> u_tets : array<Tet>;

struct Tet { p: array<u32, 4>, nr: u32, index: u32 };

struct SubTrig {
  lam: array<vec3f, 3>,
  id: u32,
}

struct ClipTetResult {
  n: u32,
  trigs: array<SubTrig, 2>,
}

fn get_tet_points(elid: u32) -> array<vec3f, 4>
{
  var p: array<vec3f, 4>;
  for(var i = 0u; i < 4u; i++) {
    for(var j = 0u; j < 3u; j++) {
      p[i][j] = vertices[u_tets[elid].p[i] * 3u + j];
    }
  }
  return p;
}

fn clipCheckOrientation(trig: SubTrig, values: array<f32, 4>) -> SubTrig {
  var result = trig;
  let n = cross(
    trig.lam[1] - trig.lam[0],
    trig.lam[2] - trig.lam[0]
  );

  let p = 1.0/3.*(trig.lam[0] + trig.lam[1] + trig.lam[2]) + n;
  let p4 = 1.0 - p.x - p.y - p.z;
  let value = p.x * values[0] + p.y * values[1] + p.z * values[2] + p4 * values[3];

  if(value<0.0) {
    result.lam[1] = trig.lam[2];
    result.lam[2] = trig.lam[1];
  }

  return result;
}

// clip tet such that the clip triangle(s) have value 0 everywhere
fn clipTet(lam: array<vec3f, 4>, values: array<f32, 4>, ei: u32) -> ClipTetResult {
    var trigs = ClipTetResult(0, array<SubTrig, 2>(SubTrig(array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0)), 0), SubTrig(array<vec3f, 3>(vec3f(0.0), vec3f(0.0), vec3f(0.0)), 0)));
    var p_pos = array<u32, 4>(0u, 0u, 0u, 0u);
    var p_neg = array<u32, 4>(0u, 0u, 0u, 0u);

    var n_pos: u32 = 0u;
    var n_neg: u32 = 0u;

    for (var i = 0u; i < 4u; i++) {
      if (values[i] > 0.0) {
            p_pos[n_pos] = i;
            n_pos++;
        } else {
            p_neg[n_neg] = i;
            n_neg++;
        }
    }

    if n_pos == 4u || n_neg == 4u {
        return trigs;
    }

    if n_pos == 3u {
        trigs.n = 1u;
        n_pos = 1u;
        n_neg = 3u;
        p_pos[3] = p_neg[0];
        p_neg[0] = p_pos[0];
        p_neg[1] = p_pos[1];
        p_neg[2] = p_pos[2];
        p_pos[0] = p_pos[3];
    }

    if n_pos == 1u {
        trigs.n = 1u;
        trigs.trigs[0].id = ei;
        for (var i = 0u; i < 3u; i++) {
            let t = values[p_pos[0] ] / (values[p_pos[0] ] - values[p_neg[i] ]);
            let lam_trig = mix(lam[p_pos[0] ], lam[p_neg[i] ], t);
            trigs.trigs[0].lam[i] = lam_trig;
        }

        trigs.trigs[0] = clipCheckOrientation(trigs.trigs[0], values);
        return trigs;
    }

  // two points before, two points behind clipping plane
  // -> two triangles
    trigs.n = 2u;
    var pairs = array<vec2u,4>(
        vec2u(p_pos[1], p_neg[0]),
        vec2u(p_pos[0], p_neg[0]),
        vec2u(p_pos[0], p_neg[1]),
        vec2u(p_pos[1], p_neg[1])
    );
    var points: array<vec3f, 4> = array<vec3f, 4>(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    for (var i = 0; i < 4; i++) {
        let p0 = pairs[i].x;
        let p1 = pairs[i].y;
        let t = values[p0 ] / (values[p0] - values[p1]);
        let lam_trig = mix(lam[p0], lam[p1], t);
        points[i] = lam_trig;
    }
    trigs.trigs[0].id = ei;
    trigs.trigs[0].lam = array(points[0], points[1], points[2]);
    trigs.trigs[1].id = ei;
    trigs.trigs[1].lam = array(points[0], points[2], points[3]);

    trigs.trigs[0] = clipCheckOrientation(trigs.trigs[0], values);
    trigs.trigs[1] = clipCheckOrientation(trigs.trigs[1], values);
    return trigs;
}

