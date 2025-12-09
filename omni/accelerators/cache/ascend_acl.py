import ctypes
import logging

libascendcl = ctypes.CDLL('libascendcl.so')

ACL_MEM_LOCATION_TYPE_HOST = 0
ACL_MEM_LOCATION_TYPE_DEVICE = 1

class aclrtMemLocation(ctypes.Structure):
    _fields_ = [
        ('id', ctypes.c_uint32),
        ('type', ctypes.c_uint32)
    ]

class aclrtMemcpyBatchAttr(ctypes.Structure):
    _fields_ = [
        ('dstLoc', aclrtMemLocation),
        ('srcLoc', aclrtMemLocation),
        ('rsv', ctypes.c_uint8 * 16)
    ]

aclrtMemcpyBatch = libascendcl.aclrtMemcpyBatch

try:
    aclrtMemcpyBatchAsync = libascendcl.aclrtMemcpyBatchAsync
except AttributeError:
    logging.warning("aclrtMemcpyBatchAsync not found in libascendcl.so, please check your AscendCL version.")

aclrtMemcpyBatch.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(aclrtMemcpyBatchAttr),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_size_t)
]
aclrtMemcpyBatch.restype = ctypes.c_int

aclrtStream = ctypes.c_void_p

ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEMCPY_DEFAULT = 4
ACL_MEMCPY_HOST_TO_BUF_TO_DEVICE = 5
ACL_MEMCPY_INNER_DEVICE_TO_DEVICE = 6
ACL_MEMCPY_INTER_DEVICE_TO_DEVICE = 7

aclrtCreateStream = libascendcl.aclrtCreateStream
aclrtCreateStream.argtypes = [ctypes.POINTER(aclrtStream)]
aclrtCreateStream.restype = ctypes.c_int

aclrtSynchronizeStream = libascendcl.aclrtSynchronizeStream
aclrtSynchronizeStream.argtypes = [aclrtStream]
aclrtSynchronizeStream.restype = ctypes.c_int

aclrtDestroyStream = libascendcl.aclrtDestroyStream
aclrtDestroyStream.argtypes = [aclrtStream]
aclrtDestroyStream.restype = ctypes.c_int

aclrtMemcpyAsync = libascendcl.aclrtMemcpyAsync
aclrtMemcpyAsync.argtypes = [
    ctypes.c_void_p,           # void *dst
    ctypes.c_size_t,           # size_t destMax
    ctypes.c_void_p,           # const void *src
    ctypes.c_size_t,           # size_t count
    ctypes.c_int,              # aclrtMemcpyKind kind
    aclrtStream                # aclrtStream stream
]
aclrtMemcpyAsync.restype = ctypes.c_int

class AscendCLStream:    
    def __init__(self):
        self._stream = aclrtStream()
    
    def create(self):
        rc = aclrtCreateStream(ctypes.byref(self._stream))
        if rc != 0:
            raise RuntimeError(f"aclrtCreateStream failed with error code: {rc}")
        return rc

    def sync(self):
        rc = aclrtSynchronizeStream(self._stream)
        if rc != 0:
            raise RuntimeError(f"aclrtSynchronizeStream failed with error code: {rc}")
        return rc
    
    @property
    def stream_ptr(self):
        return self._stream
    
    def memcpy_async(self, dst, dst_max, src, count, kind):
        rc = aclrtMemcpyAsync(dst, dst_max, src, count, kind, self._stream)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyAsync failed with error code: {rc}")
        return rc

    def memcpy_batch(self, dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails):
        rc = aclrtMemcpyBatch(dst,
                            dst_max,
                            src,
                            count,
                            batch_count,
                            attrs,
                            attrsIndex,
                            kind,
                            fails)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyBatch failed with error code: {rc}")
        return rc

    def memcpy_batch_async(self, dst, dst_max, src, count, batch_count, attrs, attrsIndex, kind, fails):
        rc = aclrtMemcpyBatchAsync(dst,
                                dst_max,
                                src,
                                count,
                                batch_count,
                                attrs,
                                attrsIndex,
                                kind, fails,
                                self._stream)
        if rc != 0:
            raise RuntimeError(f"aclrtMemcpyBatch failed with error code: {rc}")
        return rc
    
    def __del__(self):
        aclrtDestroyStream(self._stream)