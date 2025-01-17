from base64 import decode
from webgpu.utils import _is_pyodide

if not _is_pyodide:
    from IPython.display import display, HTML, Javascript

    # Load lilgui (only needed once)
    display(
        HTML("""<script src="https://cdn.jsdelivr.net/npm/lil-gui@0.20"></script>"""),
        Javascript(
            """
            function waitForElm(selector) {
        return new Promise(resolve => {
            if (document.querySelector(selector)) {
                return resolve(document.querySelector(selector));
            }

            const observer = new MutationObserver(mutations => {
                if (document.querySelector(selector)) {
                    observer.disconnect();
                    resolve(document.querySelector(selector));
                }
            });

            // If you get "parameter 1 is not of type 'Node'" error, see https://stackoverflow.com/a/77855838/492336
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        });
    }"""
        ),
    )

import ngsolve as ngs
import webgpu.jupyter as wj
from webgpu import RenderObject


def max_bounding_box(boxes):
    import numpy as np

    pmin = np.array(boxes[0][0])
    pmax = np.array(boxes[0][1])
    for b in boxes[1:]:
        pmin = np.minimum(pmin, np.array(b[0]))
        pmax = np.maximum(pmax, np.array(b[1]))
    return (pmin, pmax)


class Scene:
    def __init__(self, canvas_id, render_objects):
        self.canvas_id = canvas_id
        self.render_objects = render_objects

    def init(self, gpu):
        for obj in self.render_objects:
            obj.gpu = gpu
            obj.update()

    def Redraw(self):
        wj._run_js_code(self.data, self.canvas_id)


def DrawPyodide(scene: Scene):
    import numpy as np
    import js
    import pyodide.ffi

    objects = scene.render_objects
    gpu = objects[0].gpu
    pmin, pmax = max_bounding_box([o.get_bounding_box() for o in objects])
    gpu.input_handler.transform._center = 0.5 * (pmin + pmax)
    gpu.input_handler.transform._scale = 2 / np.linalg.norm(pmax - pmin)
    if not (pmin[2] == 0 and pmax[2] == 0):
        gpu.input_handler.transform.rotate(30, -20)
    gpu.input_handler._update_uniforms()

    def render_function(t):
        gpu.update_uniforms()
        encoder = gpu.device.createCommandEncoder()
        for obj in objects:
            obj.render(encoder)
        gpu.device.queue.submit([encoder.finish()])

    render_function = pyodide.ffi.create_proxy(render_function)
    gpu.input_handler.render_function = render_function
    js.requestAnimationFrame(render_function)


def setup(gpu, scene):
    import ngsolve_webgpu.jupyter as nwj

    scene.init(gpu)
    nwj.DrawPyodide(scene)


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
    canvas_id = wj._get_canvas_id()
    html_code = f"""
<div id="{canvas_id + '_row'}" style="display: flex; justify-content: space-between;">
    <canvas id="{canvas_id}" style="flex: 3; margin-right: 10px; border: 1px solid black; padding: 10px; height: {height}px; width: {width}px; background-color: #d0d0d0;"></canvas>
    <div id="{canvas_id + '_gui'}" style="flex: 1; margin-left: 10px; border: 1px solid black; padding: 10px;"></div>
</div>
"""
    js_code = r"""
async function draw() {{
    var gui_element = document.getElementById('{canvas_id}' + '_gui');
    console.log('gui_element =', gui_element);
    window.gui = new lil.GUI({{container: gui_element}});
    var canvas2 = document.createElement('canvas');
    console.log("canvas2 =", canvas2);
    var canvas = document.getElementById("{canvas_id}");
    console.log(canvas);
    canvas.width = {width};
    canvas.height = {height};
    canvas.style = "background-color: #d0d0d0";
    await window.webgpu_ready;
    await window.pyodide.runPythonAsync('import webgpu.jupyter; webgpu.jupyter._draw_client("{canvas_id}", "{scene}", "{assets}", globals())');
}}
draw();
    """
    render_objects = []
    if isinstance(obj, ngs.Mesh):
        mesh = obj
    if isinstance(obj, ngs.CoefficientFunction):
        if mesh is None:
            if isinstance(mesh, ngs.GridFunction):
                mesh = mesh.space.mesh
            else:
                raise ValueError("If obj is a CoefficientFunction, mesh is required.")
        from .mesh import MeshRenderObject, MeshData

        data = MeshData(mesh, obj, order=order)
        r_cf = MeshRenderObject(None, data)
        render_objects.append(r_cf)

    scene = Scene(canvas_id, render_objects)
    display(
        HTML(html_code),
        Javascript(
            js_code.format(
                canvas_id=canvas_id,
                scene=wj._encode_data(scene),
                assets=wj._encode_data(
                    {
                        "modules": {
                            "ngsolve_webgpu": wj.create_package_zip("ngsolve_webgpu")
                        },
                        "init_function": wj._encode_function(setup),
                    }
                ),
                width=width,
                height=height,
            )
        ),
    )
    return scene
