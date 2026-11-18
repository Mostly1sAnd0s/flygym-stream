# Integration verification — 2026-11-18

## Subsequent local Windows hosting check

Package downloads became available. Installed the HTTP/WebSocket test/runtime
dependencies into the existing laptop environment. Added an explicit
`serve --local-windows` option and a PowerShell launcher; default Linux EGL
requirements still apply without that option (three new tests passed).

Started a temporary localhost server with the full connectome on the RTX 5070 Ti
GPU and CPU body physics. `/healthz` reported the running GPU neural backend;
`/` served the browser page; a WebSocket client completed the protocol handshake
and received three JPEG frames. The test sent pause and stopped the server.
This verifies local HTTP and stream transport, not remote Linux EGL performance.

## Initial integration checks

Verified locally with Python 3.12, FlyGym 2.1-compatible installed APIs and
MuJoCo 3.9.0. These are local checks, not evidence of remote deployment.

- Four integration tests passed: block coupling/controller stepping,
  deterministic reset, disconnected-output parity, zero input/invalid inputs.
- Seven original Eon equation/event/decoder tests passed.
- Three original transmitter-policy tests and the original Shiu zero-input
  and timestep test passed.
- Full 157,789-neuron connectome loaded and annotation/preflight checks passed.
  One 15 ms sensory brain/body block completed on CPU with finite body state
  and 124 spikes. Repeating after reset reproduced both the per-neuron spike
  counts and full body positions exactly. This short check tests execution,
  not sustained locomotion or navigation.
- The actual Flywire scene rendered a nonconstant 160×120 RGB frame through
  the server's existing OffscreenRenderer and encoded a valid JPEG. This
  used the local Windows rendering backend, not Linux EGL/WebSocket transport.
- All Python sources compile and satisfy the existing under-500-line rule.
- Existing web assets, deployment files, app, protocol, simulation loop,
  renderer, configuration defaults and uv.lock are byte-for-byte unchanged.
  Existing project dependencies are unchanged; neural dependencies are opt-in.

The full pytest suite could not run: pytest and HTTP test dependencies were
unavailable, and PyPI DNS resolution failed both inside and outside the sandbox.
Remote GPU execution, Linux EGL, browser transport and throughput still need
checking on the GPU server. The new streaming integration has not been pushed,
deployed or connected to that computer. The original GPU neural runner was
verified previously; that is not a substitute for a remote integration test.
