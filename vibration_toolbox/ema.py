import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def frf(x, f, dt):
    r"""Return the frequency response function.

    Calculates :math:`H(i\\omega)`, and coherance of the sampled data.

    Parameters
    ----------
    x: array
        Array with the displacement data
    f: array
        Array with the force data
    dt: float
        Time step of the sampled data
    n: int
        Number of points in the fft

    Returns
    -------
    freq: array
        Driving frequencies
    mag: array
        Magnitude of the frequency response function
    ang: array
        Phase of the frequency response function
    coh: array
        Coherence function

        Plot with the frf magnitude, phase and
        coherence.

    Examples
    --------
    >>> # First we need to load the sampled data which in a .mat file
    >>> import vibration_toolbox as vtb
    >>> import scipy.io as sio
    >>> data = sio.loadmat(vtb.__path__[0] + '/data/frf_data1.mat')
    >>> #print(data)
    >>> # Data is imported as arrays. Modify then to fit our function
    >>> x = data['x']
    >>> x = x.reshape(len(x))
    >>> f = data['f']
    >>> f = f.reshape(len(f))
    >>> dt = data['dt']
    >>> dt = float(dt)
    >>> # Now we are able to call the function
    >>> freq, mag, ang, coh = vtb.frf(x, f, dt)
    >>> mag[10]
    1.018394853080...

    """
    w = np.sin(np.pi * np.arange(len(f)) / len(f))**2  # window
    # apply window
    xw = x * w
    fw = f * w
    # take ffts
    FX = np.fft.fft(xw)
    FF = np.fft.fft(fw)
    # calculate the spectral densities
    SXF = FF * np.conj(FX)
    SXX = FX * np.conj(FX)
    SFF = FF * np.conj(FF)
    # calculate the frequency response functions
    TXF = SXX / SXF

    lt = len(TXF) // 2
    freq = np.arange(lt) / (2 * lt * dt)

    TXF = TXF[:lt]
    mag = np.absolute(TXF)
    ang = np.angle(TXF) * 180 / np.pi

    coh = (np.absolute(SXF)**2) / (SXX * SFF)
    coh = np.real(coh)

    # plot H(w)
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax2)
    fig.tight_layout()

    ax1.set_title(r'$H(\omega)$ - Magnitude')
    ax2.set_title(r'$H(\omega)$ - Phase')
    ax3.set_title(r'$H(\omega)$ - Coherence')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylim(0, 2)

    ax1.semilogy(freq, mag)
    ax2.plot(freq, ang)
    ax3.plot(freq, coh[:lt])

    plt.show()

    return freq, mag, ang, coh


def plot_fft(t, time_response, ax=None, **kwargs):
    """Plot fft ot time response.

    Parameters
    ----------
    t : array
        Time array.
    time_response : array
        Array with the system's time response.
    ax : matplotlib.axes, optional
        Matplotlib axes where the amplitude will be plotted.
        If None creates a new.

    Returns
    -------
    ax : array with matplotlib.axes, optional
        Matplotlib axes array created with plt.subplots.
        Plot has frequency in rad/s and magnitude in meters peak to peak.

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> t = np.linspace(0, 10, 1000)
    >>> time_response = 2 * np.sin(40*t)
    >>> vtb.plot_fft(t, time_response)
    <AxesSubplot:title=...

    """
    if ax is None:
        ax = plt.gca()

    Ts = t[1] - t[0]  # sampling interval
    Fs = 1 / Ts  # sampling rate

    n = len(time_response)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    freq_range = (k / T)[:(n // 2)] * 2 * np.pi  # one side frequency range
    y = np.fft.fft(time_response) * 4 / n  # * 4 / n to normalize to pk-pk
    y = y[:(n // 2)]
    ax.plot(freq_range, abs(y), **kwargs)
    ax.set_xlim(freq_range[0], freq_range[-1])
    ax.set_xlabel('Freq (rad/s)')
    ax.set_ylabel('Amplitude (m - pk-pk)')

    return ax


def sdof_cf(f, TF, Fmin=None, Fmax=None):
    r"""Curve fit to a single degree of freedom FRF.

    Only one peak may exist in the segment of the FRF passed to sdofcf. No
    zeros may exist within this segment. If so, curve fitting becomes
    unreliable.

    If Fmin and Fmax are not entered, the first and last elements of TF are
    used.

    Parameters
    ----------
    f: array
        The frequency vector in Hz. Does not have to start at 0 Hz.
    TF: array
        The complex transfer function
    Fmin: int
        The minimum frequency to be used for curve fitting in the FRF
    Fmax: int
        The maximum frequency to be used for curve fitting in the FRF

    Returns
    -------
    z: double
        The damping ratio
    nf: double
        Natural frequency (Hz)
    a: double
        The numerator of the identified transfer functions

        Plot of the FRF magnitude and phase.

    Examples
    --------
    >>> # First we need to load the sampled data which is in a .mat file
    >>> import vibration_toolbox as vtb
    >>> import scipy.io as sio
    >>> data = sio.loadmat(vtb.__path__[0] + '/data/case1.mat')
    >>> #print(data)
    >>> # Data is imported as arrays. Modify then to fit our function.
    >>> TF = data['Hf_chan_2']
    >>> f = data['Freq_domain']
    >>> # Now we are able to call the function
    >>> z, nf, a = vtb.sdof_cf(f,TF,500,1000)
    >>> nf
    212.092530551...

    """
    # check fmin fmax existance
    if Fmin is None:
        inlow = 0
    else:
        inlow = Fmin

    if Fmax is None:
        inhigh = np.size(f)
    else:
        inhigh = Fmax

    if f[inlow] == 0:
        inlow = 1

    f = f[inlow:inhigh, :]
    TF = TF[inlow:inhigh, :]

    R = TF
    y = np.amax(np.abs(TF))
    cin = np.argmax(np.abs(TF))

    ll = np.size(f)

    w = f * 2 * np.pi * 1j

    w2 = w * 0
    R3 = R * 0

    for i in range(1, ll + 1):
        R3[i - 1] = np.conj(R[ll - i])
        w2[i - 1] = np.conj(w[ll - i])

    w = np.vstack((w2, w))
    R = np.vstack((R3, R))

    N = 2
    x, y = np.meshgrid(np.arange(0, N + 1), R)
    x, w2d = np.meshgrid(np.arange(0, N + 1), w)
    c = -1 * w**N * R

    aa1 = w2d[:, np.arange(0, N)] \
        ** x[:, np.arange(0, N)] \
        * y[:, np.arange(0, N)]
    aa2 = -w2d[:, np.arange(0, N + 1)] \
        ** x[:, np.arange(0, N + 1)]
    aa = np.hstack((aa1, aa2))

    aa = np.reshape(aa, [-1, 5])

    b, _, _, _ = la.lstsq(aa, c)

    b = b.flatten()
    rs = np.roots(np.array([1,
                            b[1],
                            b[0]]))
    omega = np.abs(rs[1])
    z = -1 * np.real(rs[1]) / np.abs(rs[1])
    nf = omega / 2 / np.pi

    XoF1 = np.hstack(([1 / (w - rs[0]), 1 / (w - rs[1])]))
    XoF2 = 1 / (w**0)
    XoF3 = 1 / w**2
    XoF = np.hstack((XoF1, XoF2, XoF3))

    # check if extra _ needed

    a, _, _, _ = la.lstsq(XoF, R)
    XoF = XoF[np.arange(ll, 2 * ll), :].dot(a)

    a = np.sqrt(-2 * np.imag(a[0]) * np.imag(rs[0])
                - 2 * np.real(a[0]) * np.real(rs[0]))
    Fmin = np.min(f)
    Fmax = np.max(f)
    phase = np.unwrap(np.angle(TF), np.pi, 0) * 180 / np.pi
    phase2 = np.unwrap(np.angle(XoF), np.pi, 0) * 180 / np.pi
    while phase2[cin] > 50:
        phase2 = phase2 - 360
    phased = phase2[cin] - phase[cin]
    phase = phase + np.round(phased / 360) * 360

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.tight_layout()

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.plot(f, 20 * np.log10(np.abs(XoF)), label="Identified FRF")
    ax1.plot(f, 20 * np.log10(np.abs(TF)), label="Experimental FRF")
    ax1.legend()

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.plot(f, phase2, label="Identified FRF")
    ax2.plot(f, phase, label="Experimental FRF")
    ax2.legend()

    plt.show()

    a = a[0]**2 / (2 * np.pi * nf)**2
    return z, nf, a


def mdof_cf(f, TF, Fmin=None, Fmax=None):
    """Curve fit to multiple degree of freedom FRF.

    If Fmin and Fmax are not entered, the first and last elements of TF are
    used.

    If the first column of TF is a collocated (input and output location are
    the same), then the mode shape returned is the mass normalized mode shape.
    This can then be used to generate an identified mass, damping, and
    stiffness matrix as shown in the following example.

    Parameters
    ----------
    f: array
        The frequency vector in Hz. Does not have to start at 0 Hz.
    TF: array
        The complex transfer function
    Fmin: int
        The minimum frequency to be used for curve fitting in the FRF
    Fmax: int
        The maximum frequency to be used for curve fitting in the FRF

    Returns
    -------
    z: double
        The damping ratio
    nf: double
        Natural frequency (Hz)
    u: array
        The mode shape

    Notes
    -----
    FRF are columns comprised of the FRFs presuming single input, multiple
    output z and nf are the damping ratio and natural frequency (Hz) u is the
    mode shape. Only one peak may exist in the segment of the FRF passed to
    sdofcf. No zeros may exist within this segment. If so, curve fitting
    becomes unreliable.

    Examples
    --------
    >>> # First we need to load the sampled data which is in a .mat file
    >>> import vibration_toolbox as vtb
    >>> import scipy.io as sio
    >>> data = sio.loadmat(vtb.__path__[0] + '/data/case2.mat')
    >>> #print(data)
    >>> # Data is imported as arrays. Modify then to fit our function
    >>> TF = data['Hf_chan_2']
    >>> f = data['Freq_domain']
    >>> # Now we are able to call the function
    >>> z, nf, a = vtb.mdof_cf(f,TF,500,1000)
    >>> nf
    192.59382330...

    """
    # check fmin fmax existance
    if Fmin is None:
        inlow = 0
    else:
        inlow = Fmin

    if Fmax is None:
        inhigh = np.size(f)
    else:
        inhigh = Fmax

    if f[inlow] == 0:
        inlow = 1

    f = f[inlow:inhigh, :]
    TF = TF[inlow:inhigh, :]

    R = TF.T

    U, _, _ = np.linalg.svd(R)
    T = U[:, 0]
    Hp = np.transpose(T).dot(R)
    R = np.transpose(Hp)

    ll = np.size(f)
    w = f * 2 * np.pi * 1j

    w2 = w * 0
    R3 = R * 0
    TF2 = TF * 0
    for i in range(1, ll + 1):
        R3[i - 1] = np.conj(R[ll - i])
        w2[i - 1] = np.conj(w[ll - i])
        TF2[i - 1, :] = np.conj(TF[ll - i, :])

    w = np.vstack((w2, w))
    R = np.hstack((R3, R))

    N = 2
    x, y = np.meshgrid(np.arange(0, N + 1), R)

    x, w2d = np.meshgrid(np.arange(0, N + 1), w)

    R = np.ndarray.flatten(R)
    w = np.ndarray.flatten(w)
    c = -1 * w**N * R

    aa1 = w2d[:, np.arange(0, N)] \
        ** x[:, np.arange(0, N)] \
        * y[:, np.arange(0, N)]
    aa2 = -w2d[:, np.arange(0, N + 1)] \
        ** x[:, np.arange(0, N + 1)]
    aa = np.hstack((aa1, aa2))

    b, _, _, _ = la.lstsq(aa, c)

    rs = np.roots(np.array([1,
                            b[1],
                            b[0]]))

    # irs = np.argsort(np.abs(np.imag(rs))) # necessary?

    omega = np.abs(rs[1])
    z = -1 * np.real(rs[1]) / np.abs(rs[1])
    nf = omega / 2 / np.pi

    XoF1 = 1 / ((rs[0] - w) * (rs[1] - w))

    XoF2 = 1 / (w**0)
    XoF3 = 1 / w**2

    XoF = np.vstack((XoF1, XoF2, XoF3)).T
    TF3 = np.vstack((TF2, TF))

    a, _, _, _ = la.lstsq(XoF, TF3)

    u = np.transpose(a[0, :])

    u = u / np.sqrt(np.abs(a[0, 0]))

    return z, nf, u
