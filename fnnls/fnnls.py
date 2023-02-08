# FNNLSa implementation from Rasmus Bro (1997) fnnls.m (MATLAB file exchange)
# translated to Python
# Daniel Elnatan
# Jan 22, 2018
# I believe I fixed the following fnnls algorithm by solving an indexing error.
# Ian Anthony
# August 8, 2019

from numpy import (
    zeros,
    arange,
    int64,
    float64,
    sum,
    argmax,
    nonzero,
    diag,
    min,
    abs,
    newaxis,
    finfo,
)
from scipy.linalg import solve

nu = newaxis
import numpy as np
import sys
from scipy.optimize import nnls

# machine epsilon
eps = finfo(float64).eps


def any(a):
    # assuming a vector, a
    larger_than_zero = sum(a > 0)
    if larger_than_zero:
        return True
    else:
        return False


def find_nonzero(a):
    # returns indices of nonzero elements in a
    return nonzero(a)[0]


def FNNLSa(XtX, Xty):
    tol = eps

    M, N = XtX.shape
    # initialize passive set, P. Indices where coefficient is >0
    P = zeros(N, dtype=int64)
    # and active set. Indices where coefficient is <=0
    Z = arange(N) + 1
    # working active set
    ZZ = arange(N) + 1
    # initial solution vector, x
    x = zeros(N, dtype=float64)
    # weight vector
    w = Xty - XtX @ x
    # iteration counts and parameter
    it = 0
    itmax = 30 * N

    # MAIN LOOP
    # continue as long as there are indices within the active set Z
    # or elements in inner loop active set is larger than 'tolerance'
    piter = 0
    while any(Z) and any(w[ZZ - 1] > tol):
        piter += 1
        t = argmax(w[ZZ - 1]) + 1  # find largest weight
        t = ZZ[t - 1]
        P[t - 1] = t  # move to passive set
        Z[t - 1] = 0  # remove from active set
        PP = find_nonzero(P) + 1
        ZZ = find_nonzero(Z) + 1
        NZZ = ZZ.shape

        # compute trial solution, s
        s = zeros(N, dtype=float64)

        if len(PP) == 1:
            s[PP - 1] = Xty[PP - 1] / XtX[PP - 1, PP - 1]
        else:
            s[PP - 1] = solve(XtX[PP - 1, PP[:, nu] - 1], Xty[PP - 1])
        s[ZZ - 1] = 0.0  # set active coefficients to 0

        while any(s[PP - 1] <= tol) and it < itmax:
            it = it + 1
            QQ = find_nonzero((s <= tol) * P) + 1
            alpha = min(x[QQ - 1] / (x[QQ - 1] - s[QQ - 1]))
            x = x + alpha * (s - x)
            ij = find_nonzero((abs(x) < tol) * (P != 0)) + 1
            Z[ij - 1] = ij
            P[ij - 1] = 0
            PP = find_nonzero(P) + 1
            ZZ = find_nonzero(Z) + 1
            if len(PP) == 1:
                s[PP - 1] = Xty[PP - 1] / XtX[PP - 1, PP - 1]
            else:
                s[PP - 1] = solve(XtX[PP - 1, PP[:, nu] - 1], Xty[PP - 1])
            s[ZZ - 1] = 0.0
        # assign current solution, s, to x
        x = s
        # recompute weights
        w = Xty - XtX @ x

    return x, w


XtX = np.array(
    [
        [
            9.611026155476322,
            8.87111685589437,
            8.211011366377516,
            7.621082704720055,
            7.092939797905212,
            6.619262476847601,
            6.193659100250106,
            5.810543643147376,
        ],
        [
            8.87111685589437,
            8.211011366377514,
            7.621082704720055,
            7.092939797905212,
            6.619262476847601,
            6.193659100250106,
            5.810543643147375,
            5.465029534777697,
        ],
        [
            8.211011366377516,
            7.621082704720055,
            7.092939797905212,
            6.619262476847601,
            6.193659100250106,
            5.810543643147375,
            5.465029534777699,
            5.152837915031511,
        ],
        [
            7.621082704720055,
            7.092939797905212,
            6.619262476847601,
            6.193659100250106,
            5.810543643147375,
            5.465029534777699,
            5.152837915031511,
            4.8702183082070425,
        ],
        [
            7.092939797905212,
            6.619262476847601,
            6.193659100250106,
            5.810543643147375,
            5.465029534777698,
            5.152837915031511,
            4.8702183082070425,
            4.613879995136789,
        ],
        [
            6.619262476847601,
            6.193659100250106,
            5.810543643147375,
            5.465029534777699,
            5.152837915031511,
            4.8702183082070425,
            4.613879995136789,
            4.380932606745861,
        ],
        [
            6.193659100250106,
            5.810543643147375,
            5.465029534777699,
            5.152837915031511,
            4.8702183082070425,
            4.613879995136789,
            4.380932606745861,
            4.1688346695848635,
        ],
        [
            5.810543643147376,
            5.465029534777697,
            5.152837915031511,
            4.8702183082070425,
            4.613879995136789,
            4.380932606745861,
            4.1688346695848635,
            3.975349011820167,
        ],
    ]
)
Xty = np.array(
    [
        6.292528486334608,
        5.899123987164679,
        5.54451856447887,
        5.224283191432115,
        4.934536993948907,
        4.671875931227347,
        4.433311073124163,
        4.2162151516528645,
    ]
)

print(FNNLSa(XtX, Xty), "\n")

print(nnls(XtX, Xty))
