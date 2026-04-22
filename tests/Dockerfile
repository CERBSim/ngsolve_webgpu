ARG BASE_IMAGE=webgpu-base
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir --break-system-packages ngsolve ipython git+https://github.com/CERBSim/webgpu.git

WORKDIR /app
COPY pyproject.toml .
COPY ngsolve_webgpu/ ngsolve_webgpu/
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN pip install --no-cache-dir --break-system-packages .

COPY tests/ tests/

CMD ["pytest", "tests/", "-v", "--tb=short"]