
@group(0) @binding(80) var<storage, read_write> counter_isosurface: atomic<i32>;

@compute @workgroup_size(1)
fn count_iso_triangles(@builtin(global_invocation_id) id: vec3<u32>) {
  atomicAdd(&counter_isosurface, 1);
}

@compute @workgroup_size(1)
fn write_iso_triangles(@builtin(global_invocation_id) id: vec3<u32>)
{

}
