Installation
============

Requirements
------------

* **Python 3.10+**
* **NGSolve** (see https://ngsolve.org)
* A browser with **WebGPU** support: Chrome/Edge ≥ 113, Firefox Nightly with
  ``dom.webgpu.enabled``, or Safari Technology Preview.

Install
-------

.. code-block:: bash

   pip install ngsolve_webgpu

For the latest development version:

.. code-block:: bash

   pip install git+https://github.com/CERBSim/ngsolve_webgpu.git

Building the documentation
--------------------------

The documentation embeds interactive scenes by setting
``WEBGPU_EXPORTING=1`` during the Sphinx build. A headless Chromium
(via Playwright) is launched to provide a real WebGPU device for capturing
GPU buffers.

.. code-block:: bash

   pip install sphinx nbsphinx pydata-sphinx-theme playwright
   playwright install chromium
   cd docs
   WEBGPU_EXPORTING=1 sphinx-build -b html . _build/html

Open ``_build/html/index.html`` in a WebGPU-capable browser.

Verifying WebGPU
----------------

Visit https://webgpureport.org to confirm your browser exposes a WebGPU
adapter. If no adapter is reported, the embedded canvases will display
an error message.
