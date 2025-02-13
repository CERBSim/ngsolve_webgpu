import ngsolve as ngs
import webgpu.jupyter as wj

# local_path = None # change this to local path of pyodide compiled zip files
local_path = "/home/christopher/git/source/ngsolve/dist/pyodide"

if local_path is None:
    wj.pyodide_install_packages(["ngsolve"])
else:
    def local_install(local_packages):
        packages = []
        for package in local_packages:
            with open(local_path + f"/{package}.zip", "rb") as f:
                data = f.read()
            packages.append((package, data))
        packages = wj._encode_data(packages)
        print("packages = ", packages)
        wj.run_code_in_pyodide(f"""
import shutil
from pyodide._package_loader import get_dynlibs
import pyodide_js        
from pathlib import Path
import webgpu.jupyter as wj
for package, data in wj._decode_data('{packages}'):
    with open(package + '.zip', 'wb') as f:
            f.write(data)
    import os
    print("local files = ", os.listdir('.'))
    shutil.unpack_archive(package + '.zip', '.', 'zip')
    print("after local files = ", os.listdir('.'))
    libs = get_dynlibs(package + '.zip', '.zip', Path('.'))
    print('got libs = ', libs)
    for lib in libs:
        await pyodide_js._api.loadDynlib(lib, True, [])
    import importlib
    print('import package = ', package)
    importlib.import_module(package)
""")
    local_install(["pyngcore", "netgen", "ngsolve"])
    # wj.run_code_in_pyodide("""import netgen
# print('netgen', netgen.__file__)""")
#     wj.run_code_in_pyodide("""
# import micropip
# await micropip.install('scipy')    
# import scipy""")
#     wj.run_code_in_pyodide("""import ngsolve
# print('ngsolve', ngsolve.__file__)""")

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
