import ngsolve as ngs
import webgpu.jupyter as wj

local_path = None # change this to local path of pyodide compiled zip files

if local_path is None:
    wj.pyodide_install_packages(["ngsolve"])
else:
    def local_install(package):
        from IPython.display import Javascript, display
        with open(package, "rb") as f:
            package = f.read()
        package = wj._encode_data(package)
        display(
            Javascript(
                f"""
async function install_packages() {{
    let binary_string = atob('{package}');
    const len = binary_string.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {{
        bytes[i] = binary_string.charCodeAt(i);
    }}
    await pyodide.unpackArchive(bytes, 'zip');
}}
install_packages();
"""
            ))
    local_install(local_path + "/pyngcore.zip")
    local_install(local_path + "/netgen.zip")
    wj.run_code_in_pyodide("""import netgen
print('netgen', netgen.__file__)""")
    local_install(local_path + "/ngsolve.zip")
    wj.run_code_in_pyodide("""import ngsolve
print('ngsolve', ngsolve.__file__)""")

def Draw(
    obj: ngs.CoefficientFunction | ngs.Mesh,
    mesh: ngs.Mesh | None = None,
    width=600,
    height=600,
    order: int = 2,
):
    """
    NGSolve Draw command. Draws a CoefficientFunction or a Mesh with a set of options using the NGSolve webgpu framework.

    Parameters
    ----------

    obj : ngs.CoefficientFunction | ngs.Mesh
        The CoefficientFunction or Mesh to draw.

    mesh : ngs.Mesh | None
        The mesh to draw. If obj is a CoefficientFunction, mesh is required.

    width : int
        The width of the canvas.

    height : int
        The height of the canvas.

    order : int
        The order which is used to render the CoefficientFunction. Default is 2.
    """
    # create gui before calling render
    render_objects = []
    if isinstance(obj, ngs.Mesh):
        mesh = obj
    if isinstance(obj, ngs.CoefficientFunction):
        if mesh is None:
            if isinstance(mesh, ngs.GridFunction):
                mesh = mesh.space.mesh
            else:
                raise ValueError("If obj is a CoefficientFunction, mesh is required.")
        from .mesh import CoefficientFunctionRenderObject, MeshData, FunctionData

        mesh_data = MeshData(mesh.ngmesh)
        function_data = FunctionData(mesh_data, obj, order)
        r_cf = CoefficientFunctionRenderObject(function_data)
        render_objects.append(r_cf)

    scene = wj.Scene(render_objects)
    wj.Draw(scene, width, height, modules=["ngsolve_webgpu"])
    return scene
