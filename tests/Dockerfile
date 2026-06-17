ARG BASE_IMAGE=webgpu-base
FROM ${BASE_IMAGE}

RUN git clone https://github.com/CERBSim/webgpu.git /tmp/webgpu \
    && cd /tmp/webgpu && git lfs pull \
    && pip install --no-cache-dir --break-system-packages /tmp/webgpu \
    && rm -rf /tmp/webgpu

RUN pip install --no-cache-dir --break-system-packages ipython
RUN pip install --no-cache-dir --break-system-packages anywidget
RUN pip install --no-cache-dir --break-system-packages --pre ngsolve

WORKDIR /app
COPY .git .git
COPY pyproject.toml .
COPY ngsolve_webgpu/ ngsolve_webgpu/
RUN pip install --no-cache-dir --break-system-packages --no-deps .

COPY tests/ tests/

CMD ["pytest", "tests/", "-v", "--tb=short"]
