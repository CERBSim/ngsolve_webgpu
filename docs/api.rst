API Reference
=============

.. currentmodule:: ngsolve_webgpu

The high-level :func:`~ngsolve_webgpu.jupyter.Draw` function is the easiest
entry point. The classes below let you compose custom scenes from individual
renderers.

Mesh rendering
--------------

.. autoclass:: ngsolve_webgpu.mesh.MeshData
   :members:

.. autoclass:: ngsolve_webgpu.mesh.MeshWireframe2d
.. autoclass:: ngsolve_webgpu.mesh.MeshElements2d
.. autoclass:: ngsolve_webgpu.mesh.MeshElements3d
.. autoclass:: ngsolve_webgpu.mesh.MeshSegments

Coefficient functions
---------------------

.. autoclass:: ngsolve_webgpu.cf.FunctionData
   :members:

.. autoclass:: ngsolve_webgpu.cf.CFRenderer

.. autoclass:: ngsolve_webgpu.facet_cf.FacetFunctionData
.. autoclass:: ngsolve_webgpu.facet_cf.FacetCFRenderer
.. autoclass:: ngsolve_webgpu.facet_cf.FacetCFRenderer3D

Clipping
--------

.. autoclass:: ngsolve_webgpu.clipping.ClippingCF

Vectors and isosurfaces
-----------------------

.. autoclass:: ngsolve_webgpu.vectors.SurfaceVectors
.. autoclass:: ngsolve_webgpu.vectors.ClippingVectors

Geometry
--------

.. autoclass:: ngsolve_webgpu.geometry.GeometryRenderer

Entity numbering
----------------

.. autoclass:: ngsolve_webgpu.entity_numbers.EntityNumbers

Picking
-------

.. autoclass:: ngsolve_webgpu.pick.MeshPickResult
.. autoclass:: ngsolve_webgpu.pick.GeoPickResult
.. autoclass:: ngsolve_webgpu.pick.HighlightUniforms

Symmetry
--------

.. autoclass:: ngsolve_webgpu.symmetry.Symmetry

Jupyter helpers
---------------

.. autofunction:: ngsolve_webgpu.jupyter.Draw
