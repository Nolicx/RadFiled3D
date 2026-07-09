import numpy as np


def rf3_fix_histograms_spatial_only(hist4d: np.ndarray) -> np.ndarray:
    # Korrigiert nur X,Y,Z; B bleibt unangetastet
    X, Y, Z, B = hist4d.shape
    Nvox = X * Y * Z
    perm = np.arange(Nvox).reshape((X, Y, Z), order="F").ravel(order="C")
    h = np.moveaxis(hist4d, -1, 0).reshape(B, Nvox)  # (B,Nvox)
    h = h[:, perm]  # nur räumlich permutieren
    h = h.reshape(B, X, Y, Z)
    return np.moveaxis(h, 0, -1)  # zurück zu (X,Y,Z,B)


def rf3_unfix_histograms_spatial_only(fixed4d: np.ndarray) -> np.ndarray:
    """
    Inverse of rf3_fix_histograms_spatial_only:
    restores the original (X,Y,Z,B) spatial ordering.
    """
    X, Y, Z, B = fixed4d.shape
    Nvox = X * Y * Z

    # The forward perm was: arange(N).reshape((X,Y,Z), order='F').ravel(order='C')
    # An explicit inverse is the CF permutation below (faster than argsort).
    invperm = np.arange(Nvox).reshape((X, Y, Z), order="C").ravel(order="F")

    h = np.moveaxis(fixed4d, -1, 0).reshape(B, Nvox)  # (B, Nvox)
    h = h[:, invperm]  # undo the spatial permutation
    h = h.reshape(B, X, Y, Z)
    return np.moveaxis(h, 0, -1)  # back to (X, Y, Z, B)


def rf3_fix_spatial_3d(a3d: np.ndarray) -> np.ndarray:
    X, Y, Z = a3d.shape
    return a3d.ravel(order="C").reshape((X, Y, Z), order="F")


def rf3_unfix_spatial_3d(a3d_fixed: np.ndarray) -> np.ndarray:
    """
    Inverse of rf3_fix_spatial_3d:
    restores the original (X,Y,Z) spatial ordering.
    """
    X, Y, Z = a3d_fixed.shape
    return a3d_fixed.ravel(order="F").reshape((X, Y, Z), order="C")
