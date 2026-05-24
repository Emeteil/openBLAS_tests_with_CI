import numpy as np
import ctypes, os

def test_axpy(lib):
    print("\nTesting AXPY")
    tests_passed = 0
    tests_total = 4
    
    try:
        lib.cblas_saxpy.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_saxpy.restype = None
        x = np.array([1.0, 2.0], dtype=np.float32); y = np.array([10.0, 10.0], dtype=np.float32)
        lib.cblas_saxpy(2, 2.0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("saxpy OK. Result Y:", y)
        tests_passed += 1
    except Exception as e:
        print("saxpy FAILED:", e)

    try:
        lib.cblas_daxpy.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.cblas_daxpy.restype = None
        x_d = np.array([1.0, 2.0], dtype=np.float64); y_d = np.array([10.0, 10.0], dtype=np.float64)
        lib.cblas_daxpy(2, 2.0, x_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, y_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
        print("daxpy OK. Result Y:", y_d)
        tests_passed += 1
    except Exception as e:
        print("daxpy FAILED:", e)

    try:
        lib.cblas_caxpy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.cblas_caxpy.restype = None
        x_c = np.array([1.0+1.0j, 2.0+2.0j], dtype=np.complex64); y_c = np.array([1.0+0.0j, 1.0+0.0j], dtype=np.complex64)
        alpha_c = (ctypes.c_float * 2)(2.0, 0.0)
        lib.cblas_caxpy(2, ctypes.byref(alpha_c), x_c.ctypes.data_as(ctypes.c_void_p), 1, y_c.ctypes.data_as(ctypes.c_void_p), 1)
        print("caxpy OK. Result Y:", y_c)
        tests_passed += 1
    except Exception as e:
        print("caxpy FAILED:", e)

    try:
        lib.cblas_zaxpy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.cblas_zaxpy.restype = None
        x_z = np.array([1.0+1.0j, 2.0+2.0j], dtype=np.complex128); y_z = np.array([10.0+0.0j, 10.0+0.0j], dtype=np.complex128)
        alpha_z = (ctypes.c_double * 2)(2.0, 0.0)
        lib.cblas_zaxpy(2, ctypes.byref(alpha_z), x_z.ctypes.data_as(ctypes.c_void_p), 1, y_z.ctypes.data_as(ctypes.c_void_p), 1)
        print("zaxpy OK. Result Y:", y_z)
        tests_passed += 1
    except Exception as e:
        print("zaxpy FAILED:", e)

    return tests_passed, tests_total

def test_scal(lib):
    print("\nTesting SCAL")
    tests_passed = 0
    tests_total = 3
    
    try:
        lib.cblas_sscal.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_sscal.restype = None
        x = np.array([1.0, 2.0], dtype=np.float32)
        lib.cblas_sscal(2, 3.0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("sscal OK. Result X:", x)
        tests_passed += 1
    except Exception as e:
        print("sscal FAILED:", e)

    try:
        lib.cblas_dscal.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.cblas_dscal.restype = None
        x_d = np.array([1.0, 2.0], dtype=np.float64)
        lib.cblas_dscal(2, 3.0, x_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
        print("dscal OK. Result X:", x_d)
        tests_passed += 1
    except Exception as e:
        print("dscal FAILED:", e)

    try:
        lib.cblas_cscal.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.cblas_cscal.restype = None
        x_c = np.array([1.0+1.0j, 2.0+2.0j], dtype=np.complex64)
        alpha_c = (ctypes.c_float * 2)(2.0, 0.0)
        lib.cblas_cscal(2, ctypes.byref(alpha_c), x_c.ctypes.data_as(ctypes.c_void_p), 1)
        print("cscal OK. Result X:", x_c)
        tests_passed += 1
    except Exception as e:
        print("cscal FAILED:", e)

    return tests_passed, tests_total

def test_copy(lib):
    print("\nTesting COPY")
    tests_passed = 0
    tests_total = 2
    
    try:
        lib.cblas_scopy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_scopy.restype = None
        x = np.array([1.0, 2.0], dtype=np.float32); y = np.zeros(2, dtype=np.float32)
        lib.cblas_scopy(2, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("scopy OK. Result Y:", y)
        tests_passed += 1
    except Exception as e:
        print("scopy FAILED:", e)

    try:
        lib.cblas_dcopy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.cblas_dcopy.restype = None
        x_d = np.array([1.0, 2.0], dtype=np.float64); y_d = np.zeros(2, dtype=np.float64)
        lib.cblas_dcopy(2, x_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, y_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
        print("dcopy OK. Result Y:", y_d)
        tests_passed += 1
    except Exception as e:
        print("dcopy FAILED:", e)

    return tests_passed, tests_total

def test_swap(lib):
    print("\nTesting SWAP")
    tests_passed = 0
    tests_total = 2
    
    try:
        lib.cblas_sswap.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_sswap.restype = None
        x = np.array([1.0, 2.0], dtype=np.float32); y = np.array([3.0, 4.0], dtype=np.float32)
        lib.cblas_sswap(2, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("sswap OK. Result X:", x, "Y:", y)
        tests_passed += 1
    except Exception as e:
        print("sswap FAILED:", e)

    try:
        lib.cblas_zswap.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.cblas_zswap.restype = None
        x_z = np.array([1.0+1.0j, 2.0+2.0j], dtype=np.complex128); y_z = np.array([3.0+3.0j, 4.0+4.0j], dtype=np.complex128)
        lib.cblas_zswap(2, x_z.ctypes.data_as(ctypes.c_void_p), 1, y_z.ctypes.data_as(ctypes.c_void_p), 1)
        print("zswap OK. Result X:", x_z, "Y:", y_z)
        tests_passed += 1
    except Exception as e:
        print("zswap FAILED:", e)

    return tests_passed, tests_total

def test_dot(lib):
    print("\nTesting DOT")
    tests_passed = 0
    tests_total = 5
    
    try:
        lib.cblas_sdot.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_sdot.restype = ctypes.c_float
        x = np.array([1.0, 2.0], dtype=np.float32); y = np.array([3.0, 4.0], dtype=np.float32)
        res = lib.cblas_sdot(2, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("sdot OK. Result:", res)
        tests_passed += 1
    except Exception as e:
        print("sdot FAILED:", e)

    try:
        lib.cblas_ddot.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.cblas_ddot.restype = ctypes.c_double
        x_d = np.array([1.0, 2.0], dtype=np.float64); y_d = np.array([3.0, 4.0], dtype=np.float64)
        res_d = lib.cblas_ddot(2, x_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1, y_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
        print("ddot OK. Result:", res_d)
        tests_passed += 1
    except Exception as e:
        print("ddot FAILED:", e)

    try:
        lib.cblas_sdsdot.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_sdsdot.restype = ctypes.c_float
        x = np.array([1.0, 2.0], dtype=np.float32); y = np.array([3.0, 4.0], dtype=np.float32)
        res_sds = lib.cblas_sdsdot(2, 1.0, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("sdsdot OK. Result:", res_sds)
        tests_passed += 1
    except Exception as e:
        print("sdsdot FAILED:", e)

    try:
        lib.cblas_cdotu_sub.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.cblas_cdotu_sub.restype = None
        x_c = np.array([1.0+1.0j, 1.0+1.0j], dtype=np.complex64); y_c = np.array([2.0+1.0j, 2.0+1.0j], dtype=np.complex64)
        res_c = np.zeros(1, dtype=np.complex64)
        lib.cblas_cdotu_sub(2, x_c.ctypes.data_as(ctypes.c_void_p), 1, y_c.ctypes.data_as(ctypes.c_void_p), 1, res_c.ctypes.data_as(ctypes.c_void_p))
        print("cdotu_sub OK. Result:", res_c[0])
        tests_passed += 1
    except Exception as e:
        print("cdotu_sub FAILED:", e)
    
    try:
        lib.cblas_cdotc_sub.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.cblas_cdotc_sub.restype = None
        x_c = np.array([1.0+1.0j, 1.0+1.0j], dtype=np.complex64); y_c = np.array([2.0+1.0j, 2.0+1.0j], dtype=np.complex64)
        res_cc = np.zeros(1, dtype=np.complex64)
        lib.cblas_cdotc_sub(2, x_c.ctypes.data_as(ctypes.c_void_p), 1, y_c.ctypes.data_as(ctypes.c_void_p), 1, res_cc.ctypes.data_as(ctypes.c_void_p))
        print("cdotc_sub OK. Result:", res_cc[0])
        tests_passed += 1
    except Exception as e:
        print("cdotc_sub FAILED:", e)

    return tests_passed, tests_total

def test_nrm2(lib):
    print("\nTesting NRM2")
    tests_passed = 0
    tests_total = 2
    
    try:
        lib.cblas_snrm2.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_snrm2.restype = ctypes.c_float
        x = np.array([3.0, -4.0], dtype=np.float32)
        res = lib.cblas_snrm2(2, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("snrm2 OK. Result:", res)
        tests_passed += 1
    except Exception as e:
        print("snrm2 FAILED:", e)

    try:
        lib.cblas_dnrm2.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        lib.cblas_dnrm2.restype = ctypes.c_double
        x_d = np.array([3.0, -4.0], dtype=np.float64)
        res_d = lib.cblas_dnrm2(2, x_d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 1)
        print("dnrm2 OK. Result:", res_d)
        tests_passed += 1
    except Exception as e:
        print("dnrm2 FAILED:", e)

    return tests_passed, tests_total

def test_asum(lib):
    print("\nTesting ASUM")
    tests_passed = 0
    tests_total = 2
    
    try:
        lib.cblas_sasum.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_sasum.restype = ctypes.c_float
        x = np.array([3.0, -4.0], dtype=np.float32)
        res = lib.cblas_sasum(2, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("sasum OK. Result:", res)
        tests_passed += 1
    except Exception as e: print("sasum FAILED:", e)

    try:
        lib.cblas_scasum.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.cblas_scasum.restype = ctypes.c_float
        x_c = np.array([1.0+1.0j, 1.0+1.0j], dtype=np.complex64)
        res_c = lib.cblas_scasum(2, x_c.ctypes.data_as(ctypes.c_void_p), 1)
        print("scasum OK. Result:", res_c)
        tests_passed += 1
    except Exception as e:
        print("scasum FAILED:", e)

    return tests_passed, tests_total

def test_amax(lib):
    print("\nTesting AMAX")
    tests_passed = 0
    tests_total = 2
    
    try:
        lib.cblas_isamax.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        lib.cblas_isamax.restype = ctypes.c_int
        x = np.array([1.0, -5.0, 2.0], dtype=np.float32)
        res = lib.cblas_isamax(3, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1)
        print("isamax OK. Result (index):", res)
        tests_passed += 1
    except Exception as e:
        print("isamax FAILED:", e)

    try:
        lib.cblas_izamax.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        lib.cblas_izamax.restype = ctypes.c_int
        x_z = np.array([1.0+0j, 0.0-5.0j, 2.0+0j], dtype=np.complex128)
        res_z = lib.cblas_izamax(3, x_z.ctypes.data_as(ctypes.c_void_p), 1)
        print("izamax OK. Result (index):", res_z)
        tests_passed += 1
    except Exception as e:
        print("izamax FAILED:", e)

    return tests_passed, tests_total

def test_rot(lib):
    print("\nTesting ROT")
    tests_passed = 0
    tests_total = 1
    
    try:
        lib.cblas_srot.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_float]
        lib.cblas_srot.restype = None
        x = np.array([1.0, 2.0], dtype=np.float32); y = np.array([3.0, 4.0], dtype=np.float32)
        lib.cblas_srot(2, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 1, 0.0, 1.0)
        print("srot OK. X:", x, "Y:", y)
        tests_passed += 1
    except Exception as e:
        print("srot FAILED:", e)

    return tests_passed, tests_total

if __name__ == "__main__":
    try:
        lib = ctypes.CDLL(os.path.abspath("build/OpenBLAS/lib/libopenblas.so"))
        # lib = ctypes.CDLL(os.path.abspath("build/broken_libopenblas.so"))
        
        all_passed = 0
        all_total = 0
        
        test_funcs = [
            test_axpy,
            test_scal,
            test_copy,
            test_swap,
            test_dot,
            test_nrm2,
            test_asum,
            test_amax,
            test_rot
        ]
        
        for func in test_funcs:
            passed, total = func(lib)
            all_passed += passed
            all_total += total
            
        print(f"\nTotal Tests Passed: {all_passed} / {all_total}")
    except Exception as e:
        print("Error:", e)