struct VertexOutput1d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: f32,
  @location(2) @interpolate(flat) id: u32,
};

struct VertexOutput2d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec2<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

struct VertexOutput3d {
  @builtin(position) fragPosition: vec4<f32>,
  @location(0) p: vec3<f32>,
  @location(1) lam: vec3<f32>,
  @location(2) @interpolate(flat) id: u32,
  @location(3) n: vec3<f32>,
};

@vertex
fn vertexEdgeP1(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) edgeId: u32) -> VertexOutput1d {
    let edge = edges_p1[edgeId];
    var p: vec3<f32> = vec3<f32>(edge.p[3 * vertexId], edge.p[3 * vertexId + 1], edge.p[3 * vertexId + 2]);

    var lam: f32 = 0.0;
    if vertexId == 0 {
        lam = 1.0;
    }

    var position = cameraMapPoint(p);
    return VertexOutput1d(position, p, lam, edgeId);
}

fn calcTrig(p: array<vec3<f32>, 3>, vertexId: u32, trigId: u32) -> VertexOutput2d {
    var lam: vec3<f32> = vec3<f32>(0.);
    lam[vertexId] = 1.0;

    let position = cameraMapPoint(p[vertexId]);
    let normal = cross(p[1] - p[0], p[2] - p[0]);

    return VertexOutput2d(position, p[vertexId], lam.xy, trigId, normal);
}

@vertex
fn vertexTrigP1Indexed(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) trigId: u32) -> VertexOutput2d {
    var vid = 3 * vec3u(
        trigs[4 * trigId + 0],
        trigs[4 * trigId + 1],
        trigs[4 * trigId + 2]
    );

    var p = array<vec3<f32>, 3>(
        vec3<f32>(vertices[vid[0] ], vertices[vid[0] + 1], vertices[vid[0] + 2]),
        vec3<f32>(vertices[vid[1] ], vertices[vid[1] + 1], vertices[vid[1] + 2]),
        vec3<f32>(vertices[vid[2] ], vertices[vid[2] + 1], vertices[vid[2] + 2])
    );
    return calcTrig(p, vertexId, trigId);
}


@fragment
fn fragmentTrig(input: VertexOutput2d) -> @location(0) vec4<f32> {
    checkClipping(input.p);
    let p = &trig_function_values;
    let value = evalTrig(p, input.id, 0u, input.lam);
    return getColor(value);
}

@fragment
fn fragmentEdge(@location(0) p: vec3<f32>) -> @location(0) vec4<f32> {
    checkClipping(p);
    return vec4<f32>(0, 0, 0, 1.0);
}
