# from Lucas T. Tepper (used in Tepper et al Journal of Chemical Theory and Computation 2024)

import numpy as np
import mkl_fft


def mkl_res_conj_inplace(x, real_type, cmplx_type):
    """ The resulting array of the intel_mkl.fft, broadcasting along
    the first dimension, writes the results in the form Axis 0: n_ffts
    Axis1: Real(Nyquist freq), half the spectrum, Real(zero freq).
    This is possible due to the hermetian spectrum of real data.
    To complex conjugate this we want to conj all but the
    first and last element.
    Arguments:
        real valued 2d np.ndarray, output of mkl.rfft
        real_type (np.float32, np.float64): Type of the real input data
        cmplx_type (np.complex64, np.complex128): Corresponding complex dtype to input type
    Returns:
        pointer to input array """

    for i in range(len(x)):
        x[i, 1:-1] = np.conj(x[i, 1:-1].view(cmplx_type)).view(real_type)
    return x


def mkl_res_mult_inplace(x, y, real_type, cmplx_type):
    """ The resulting array of the intel_mkl.fft, broadcasting along
    the first dimension, writes the results in the form Axis 0: n_ffts
    Axis1: Real(Nyquist freq), half the spectrum, Real(zero freq).
    This is possible due to the hermetian spectrum of real data.
    To multiply two arrays, we need to do complex multiplication
    for spectrum and pointwise for first and last elem.
    Arguments:
        (x) real valued 2d np.ndarray, output of mkl.rfft
        (y) real valued 2d np.ndarray, output of mkl.rfft
        real_type (np.float32, np.float64): Type of the real data the spectrum belongs to
        cmplx_type (np.complex64, np.complex128): Corresponding complex dtype to real type
    Returns:
        (x) real valued np.ndarray, output of mkl.rfft,
            written in input x """

    for i in range(len(x)):
        x[i, 1:-1] = (x[i, 1:-1].view(cmplx_type) * y[i, 1:-1].view(cmplx_type)).view(real_type)
    x[:, 0] *= y[:, 0]
    x[:, -1] *= y[:, -1]
    return x


def correlation(a_input, b_input=None, trunc=None):
    """ Calculates correlation via FFT. Only tested for 1d data.
    Zero pads the data. Zero padding twice makes no difference,
    so we do not check if it has been done already. We check
    if we compute ACF by id(a) == id(b) and only single fft
    in that case. If trunc not provided,returns len(a) points,
    else trunc. Only works for real data, as it uses mkl.rfft.
    Arguments:
        a (np.ndarray, real): input 1
        b (np.ndarray, real): input 2
        trunc (int): number of out elements to keep
    Returns (np.ndarray): correlation. Output has the same number
        of dimensions as the input. """

    if np.any(b_input != None):
        assert a_input.dtype == b_input.dtype
        assert isinstance(b_input, np.ndarray)
        assert a_input.shape == b_input.shape
        assert np.isreal(b_input.all())
    assert isinstance(a_input, np.ndarray)
    assert np.isreal(a_input.all())
    assert isinstance(trunc, (int, type(None)))

    # check dtype
    if a_input.dtype == np.float32:
        real_type, cmplx_type = np.float32, np.complex64
    else:
        real_type, cmplx_type = np.float64, np.complex128
    # The code assumes 2d shape, reshape from 1d, maintain input var for later
    if a_input.ndim == 1:
        reshaped = True
        a = a_input.reshape(1, -1)
        if np.any(b_input != None):
            b = b_input.reshape(1, -1)
    else:
        reshaped = False
        a, b = a_input, b_input
    # store shape of data for later
    ndim, len_a = a.shape
    if not trunc:
        trunc = len_a
    trunc = min(trunc, len_a)

    a = np.ascontiguousarray(np.concatenate([a, np.zeros_like(a)], axis=1))
    a = mkl_fft.rfft(a, n=None, axis=-1, overwrite_x=True).reshape(ndim, -1)
    #if id(a_input) == id(b_input):
    if np.all(b_input == None):
        a1 = mkl_res_conj_inplace(np.copy(a), real_type, cmplx_type)
        a = mkl_res_mult_inplace(a1, a, real_type, cmplx_type)
    else:
        if id(a_input) == id(b_input):
            a1 = mkl_res_conj_inplace(np.copy(a), real_type, cmplx_type)
            a = mkl_res_mult_inplace(a1, a, real_type, cmplx_type)
        else:
            b = np.ascontiguousarray(np.concatenate([b, np.zeros_like(b)], axis=1))
            b = mkl_fft.rfft(b, n=None, axis=-1, overwrite_x=True).reshape(ndim, -1)
            a = mkl_res_conj_inplace(a, real_type, cmplx_type)
            a = mkl_res_mult_inplace(a, b, real_type, cmplx_type)

    a = mkl_fft.irfft(a, n=None, axis=-1, overwrite_x=True).reshape(ndim, -1)[:, :trunc]
    a /= np.linspace(len_a, len_a - trunc + 1, trunc).reshape(1, -1)
    if reshaped:
        return a.flatten()
    return a
