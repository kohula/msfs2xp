"""
Optional, auto-detected GPU batch path for the GPU fork.

Uses pyopencl rather than CUDA-only libraries (cupy, torch's CUDA build) or
torch-directml (no longer published at all) specifically so this works the
same way across NVIDIA, AMD, and Intel GPUs -- through each vendor's normal
OpenCL driver, no CUDA/ROCm toolkit required.

Everything here is defensive by design: import, platform/device discovery,
context creation, and every kernel dispatch are all wrapped so a missing
install, a missing GPU, or a driver hiccup never breaks the pipeline --
callers always get a correct numpy result back, just computed on the CPU
instead. HAS_GPU / describe() exist purely so main.py can log which mode is
actually active; nothing should ever branch on them for correctness.

The one place this is actually wired in (mesh_convert/convert.py's per-model
world-space vertex/normal transform) only takes this path for meshes above
a vertex-count threshold -- most MSFS scenery meshes are small enough that
GPU dispatch/transfer overhead would cost more than the plain numpy
matmul saves.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Below this many vertices, GPU dispatch overhead isn't worth it -- stay on
# the existing numpy path, which is also the exact/float64 one.
GPU_VERTEX_THRESHOLD = 20_000

_lock = threading.Lock()
_initialized = False
_ctx = None
_queue = None
_program = None
_kernel = None
HAS_GPU = False
DEVICE_NAME = None

_TRANSFORM_KERNEL_SRC = """
__kernel void transform_points(__global const float4 *points_h,
                                __global const float *matrix,
                                __global float4 *out)
{
    int gid = get_global_id(0);
    float4 p = points_h[gid];
    float4 row0 = (float4)(matrix[0], matrix[1], matrix[2], matrix[3]);
    float4 row1 = (float4)(matrix[4], matrix[5], matrix[6], matrix[7]);
    float4 row2 = (float4)(matrix[8], matrix[9], matrix[10], matrix[11]);
    out[gid] = (float4)(dot(row0, p), dot(row1, p), dot(row2, p), 0.0f);
}
"""


def _ensure_initialized():
    """Lazy, one-time, best-effort OpenCL setup. Safe to call repeatedly --
    only does real work once per process."""
    global _initialized, _ctx, _queue, _program, _kernel, HAS_GPU, DEVICE_NAME
    with _lock:
        if _initialized:
            return
        _initialized = True
        try:
            import pyopencl as cl
        except ImportError:
            return

        try:
            device = None
            for platform in cl.get_platforms():
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    device = devices[0]
                    break
            if device is None:
                return

            ctx = cl.Context([device])
            queue = cl.CommandQueue(ctx)
            program = cl.Program(ctx, _TRANSFORM_KERNEL_SRC).build()
            # Retrieve the Kernel object once and reuse it -- fetching it
            # fresh via program.transform_points(...) on every dispatch
            # (as opposed to program.transform_points, the attribute)
            # creates a brand new Kernel each time, which pyopencl warns
            # is expensive when done repeatedly.
            kernel = cl.Kernel(program, "transform_points")

            _ctx, _queue, _program, _kernel = ctx, queue, program, kernel
            DEVICE_NAME = f"{device.name.strip()} ({platform.name.strip()})"
            HAS_GPU = True
        except Exception as e:
            logger.info(f"OpenCL device found but initialization failed, staying on CPU: {e}")
            _ctx = _queue = _program = _kernel = None
            HAS_GPU = False
            DEVICE_NAME = None


def describe():
    """One-line status string for the pipeline log at startup."""
    _ensure_initialized()
    if HAS_GPU:
        return f"GPU acceleration: {DEVICE_NAME} via OpenCL (used for large-mesh transforms only)"
    return "GPU acceleration unavailable -- running fully multiprocessed on CPU"


def transform_points_gpu(points, matrix4x4):
    """points: (N, 3) array-like. matrix4x4: (4, 4) array-like, applied as
    matrix @ [x, y, z, 1]. Returns an (N, 3) float64 ndarray, or None if the
    GPU path isn't available/usable right now -- callers must fall back to
    numpy on None, this function never raises."""
    _ensure_initialized()
    if not HAS_GPU:
        return None

    try:
        import pyopencl as cl
        import pyopencl.array as cl_array

        pts = np.asarray(points, dtype=np.float32)
        n = pts.shape[0]
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)

        points_h = np.empty((n, 4), dtype=np.float32)
        points_h[:, :3] = pts
        points_h[:, 3] = 1.0

        mat = np.asarray(matrix4x4, dtype=np.float32).reshape(-1)

        points_buf = cl_array.to_device(_queue, points_h)
        matrix_buf = cl_array.to_device(_queue, mat)
        out_buf = cl_array.empty(_queue, (n, 4), dtype=np.float32)

        _kernel.set_args(points_buf.data, matrix_buf.data, out_buf.data)
        cl.enqueue_nd_range_kernel(_queue, _kernel, (n,), None)
        out = out_buf.get()
        return out[:, :3].astype(np.float64)
    except Exception as e:
        logger.info(f"GPU transform failed, falling back to CPU for this mesh: {e}")
        return None
