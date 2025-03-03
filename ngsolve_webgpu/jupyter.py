import ngsolve as ngs
import webgpu.jupyter as wj

_local_path = None  # change this to local path of pyodide compiled zip files

if not wj._is_pyodide:
    from IPython.display import Javascript, display

    def run_on_pyodide_ready(code):
        display(
            Javascript(
                f"""
function waitTillPyodideReady() {{
        window.webgpu_ready = new Promise((resolve, reject) => {{
            if(window.pyodide_ready === undefined) {{
               window.setTimeout(waitTillPyodideReady, 100);
            }} else {{
                window.pyodide_ready.then(() => {{
                window.pyodide.runPythonAsync(`{code}`).then(() => {{
                    resolve();
                }});
                }});
            }}
        }});
}}
waitTillPyodideReady();
"""
            )
        )

    if _local_path is None:
        run_on_pyodide_ready(
            """
        _NGSOLVE_BASE_URL = "https://ngsolve.org/files/pyodide-0.27.2/master/"
        print("run code")
        import micropip
        from pyodide.http import pyfetch
        import pyodide_js
        from pyodide._package_loader import unpack_buffer

        await micropip.install("scipy")
        import scipy

        for module in ["pyngcore", "netgen", "ngsolve"]:
            response = await pyfetch(f"{_NGSOLVE_BASE_URL}/{module}.zip")
            data = await response.buffer()
            dynlibs = list(
                unpack_buffer(
                    data, format="zip", filename=f"{module}.zip", calculate_dynlibs=True
                )
            )
            for lib in dynlibs:
                await pyodide_js._api.loadDynlib(lib, True, [])
            print("loaded ", module)
        """
        )
    else:

        def local_install(local_packages):
            packages = []
            for package in local_packages:
                with open(_local_path + f"/{package}.zip", "rb") as f:
                    data = f.read()
                packages.append((package, data))
            packages = wj._encode_data(packages)
            run_on_pyodide_ready(
                f"""
    import shutil
    from pyodide._package_loader import get_dynlibs
    import pyodide_js
    from pathlib import Path
    import webgpu.jupyter as wj
    import micropip
    await micropip.install('scipy')
    import scipy
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
    """
            )

        local_install(["pyngcore", "netgen", "ngsolve"])


def Draw(
    obj: ngs.CoefficientFunction | ngs.Mesh,
    mesh: ngs.Mesh | None = None,
    name: str = None,
    width=600,
    height=600,
    order: int = 2,
    vectors=None,
    **kwargs,
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
    if isinstance(obj, ngs.Mesh | ngs.Region):
        mesh = obj
        from .mesh import MeshData, Mesh2dElementsRenderer, Mesh2dWireframeRenderer

        mesh_data = MeshData(mesh)
        m2d = Mesh2dElementsRenderer(mesh_data)
        wf = Mesh2dWireframeRenderer(mesh_data)
        render_objects.append(m2d)
        render_objects.append(wf)
    if isinstance(obj, ngs.CoefficientFunction):
        if mesh is None:
            if isinstance(mesh, ngs.GridFunction):
                mesh = mesh.space.mesh
            else:
                raise ValueError("If obj is a CoefficientFunction, mesh is required.")
        from .cf import CoefficientFunctionRenderObject, FunctionData
        from .mesh import MeshData, Mesh2dWireframeRenderer

        mesh_data = MeshData(mesh)
        function_data = FunctionData(mesh_data, obj, order)
        r_cf = CoefficientFunctionRenderObject(function_data)
        wf = Mesh2dWireframeRenderer(mesh_data)
        render_objects.append(r_cf)
        render_objects.append(wf)
        render_objects.append(r_cf.colormap)
        if vectors:
            options = vectors if isinstance(vectors, dict) else {}
            if mesh.dim != 2:
                raise ValueError("Vectors currently only implemented on 2d meshes")
            from .cf import VectorCFRenderer

            vcf = VectorCFRenderer(obj, mesh, **options)
            vcf.colormap = r_cf.colormap
            render_objects.append(vcf)

    scene = wj.Scene(render_objects)
    scene = wj.Draw(scene, width, height, modules=["ngsolve_webgpu"])
    for r in render_objects:
        r.add_options_to_gui(scene.gui)
    return scene
