import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
from scipy.optimize import newton_krylov

try:
    from IPython.display import clear_output, display, HTML
    from ipywidgets.widgets.interaction import interact, interactive
except ImportError:
    print('Interactive iPython tools will not work without IPython.display \
          and ipywidgets installed.')


def _in_ipynb():
    try:
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except NameError:
        return False


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (10, 6)


def euler_beam_modes(n=10, bctype=3, npoints=2001,
                     beamparams=np.array([7.31e10, 8.4375e-09,
                                          2747.0, 4.5e-04, 0.4])):
    """Mode shapes and natural frequencies of Euler-Bernoulli beam.

    Parameters
    ----------
    n: int, numpy array
        highest mode number or array of mode numbers to return
    bctype: int
        bctype = 1 free-free
        bctype = 2 clamped-free
        bctype = 3 clamped-pinned
        bctype = 4 clamped-sliding
        bctype = 5 clamped-clamped
        bctype = 6 pinned-pinned
    beamparams: numpy array
        E, I, rho, A, L,
        Young's modulus, second moment of area, density, cross section area,
        length of beam
    npoints: int
        number of points for returned mode shape array

    Returns
    -------
    omega_n: numpy array
        array of natural frequencies
    x: numpy array
        x coordinate
    U: numpy array
        mass normalized mode shape

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb
    >>> omega_n, x, U = vtb.euler_beam_modes(n=1)
    >>> plt.figure()
    <matplotlib.figure...>
    >>> plt.plot(x,U)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('x (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('Displacement (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.title('Mode 1')
    <matplotlib.text.Text object at ...>
    >>> plt.grid('on')
    """

    E = beamparams[0]
    I = beamparams[1]
    rho = beamparams[2]
    A = beamparams[3]
    L = beamparams[4]
    if isinstance(n, int):
        ln = n
        n = np.arange(n) + 1
    else:
        ln = len(n)

    # len=[0:(1/(npoints-1)):1]';  %Normalized length of the beam
    x_normed = np.linspace(0, 1, npoints, endpoint=True)
    x = x_normed * L
    # Determine natural frequencies and mode shapes depending on the
    # boundary condition.
    # Mass simplification. The following was arange_(1,length_(n)).reshape(-1)
    mode_num_range = np.arange(0, ln)
    Bnl = np.empty(ln)
    w = np.empty(ln)
    U = np.empty([npoints, ln])

    if bctype == 1:
        desc = 'Free-Free '
        Bnllow = np.array((0, 0, 4.73004074486, 7.8532046241,
                           10.995607838, 14.1371654913, 17.2787596574))
        for i in mode_num_range:
            if n[i] > 7:
                Bnl[i] = (2 * n[i] - 3) * np.pi / 2
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            if n[i] == 1:
                w[i] = 0
                U[:, i] = 1 + x_normed * 0
            elif n[i] == 2:
                w[i] = 0
                U[:, i] = x_normed - 0.5
            else:
                sig = (np.cosh(Bnl[i]) - np.cos(Bnl[i])) / \
                      (np.sinh(Bnl[i]) - np.sin(Bnl[i]))
                w[i] = (Bnl[i] ** 2) * np.sqrt(E * I / (rho * A * L ** 4))
                b = Bnl[i] * x_normed
                U[:, i] = np.cosh(b) + np.cos(b) - sig * \
                    (np.sinh(b) + np.sin(b))
    elif bctype == 2:
        desc = 'Clamped-Free '
        Bnllow = np.array((1.88, 4.69, 7.85, 10.99, 14.14))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (2 * n[i] - 1) * np.pi / 2
            else:
                Bnl[i] = Bnllow[i]

        for i in mode_num_range:
            sig = (np.sinh(Bnl[i]) - np.sin(Bnl[i])) / \
                  (np.cosh(Bnl[i]) - np.cos(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * np.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * x_normed
            # plt.plot(x,(sp.cosh(b) - sp.cos(b) -
            # sig * (sp.sinh(b) - sp.sin(b))))
            U[:, i] = np.cosh(b) - np.cos(b) - sig * (np.sinh(b) - np.sin(b))

    elif bctype == 3:
        desc = 'Clamped-Pinned '
        Bnllow = np.array((3.93, 7.07, 10.21, 13.35, 16.49))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (4 * n[i] + 1) * np.pi / 4
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            sig = (np.cosh(Bnl[i]) - np.cos(Bnl[i])) / \
                  (np.sinh(Bnl[i]) - np.sin(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * np.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * x_normed
            U[:, i] = np.cosh(b) - np.cos(b) - sig * (np.sinh(b) - np.sin(b))
    elif bctype == 4:
        desc = 'Clamped-Sliding '
        Bnllow = np.array((2.37, 5.5, 8.64, 11.78, 14.92))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (4 * n[i] - 1) * np.pi / 4
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            sig = (np.sinh(Bnl[i]) + np.sin(Bnl[i])) / \
                  (np.cosh(Bnl[i]) - np.cos(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * np.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * x_normed
            U[:, i] = np.cosh(b) - np.cos(b) - sig * (np.sinh(b) - np.sin(b))
    elif bctype == 5:
        desc = 'Clamped-Clamped '
        Bnllow = np.array((4.73, 7.85, 11, 14.14, 17.28))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (2 * n[i] + 1) * np.pi / 2
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            sig = (np.cosh(Bnl[i]) - np.cos(Bnl[i])) / \
                  (np.sinh(Bnl[i]) - np.sin(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * np.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * x_normed
            U[:, i] = np.cosh(b) - np.cos(b) - sig * (np.sinh(b) - np.sin(b))
    elif bctype == 6:
        desc = 'Pinned-Pinned '
        for i in mode_num_range:
            Bnl[i] = n[i] * np.pi
            w[i] = (Bnl[i] ** 2) * np.sqrt(E * I / (rho * A * L ** 4))
            U[:, i] = np.sin(Bnl[i] * x_normed)

    # Mass Normalization of mode shapes
    for i in mode_num_range:
        U[:, i] = U[:, i] / np.sqrt(np.dot(U[:, i], U[:, i]) * rho * A * L)

    omega_n = w
    return omega_n, x, U


def euler_beam_frf(xin=0.22, xout=0.32, fmin=0.0, fmax=1000.0, zeta=0.02,
                   bctype=2, npoints=2001,
                   beamparams=np.array([7.31e10, 1 / 12 * 0.03 * .015 ** 3,
                                        2747.0, .015 * 0.03, 0.4])):
    """Frequency response function fo Euler-Bernoulli beam.

    Parameters
    ----------
    xin: float
        location of applied force
    xout: float
        location of displacement sensor
    fmin: float
        lowest frequency of interest
    fmax: float
        highest frequency of interest
    zeta: float
        damping ratio
    bctype: int
        bctype = 1 free-free
        bctype = 2 clamped-free
        bctype = 3 clamped-pinned
        bctype = 4 clamped-sliding
        bctype = 5 clamped-clamped
        bctype = 6 pinned-pinned
    beamparams: numpy array
        E, I, rho, A, L,
        Young's modulus, second moment of area, density, cross section area,
        length of beam
    npoints: int
        number of points for returned mode shape array

    Returns
    -------
    fout: numpy array
        array of driving frequencies (Hz)
    H: numpy array
        Frequency Response Function

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb
    >>> _, _ = vtb.euler_beam_frf()

    """

    E = beamparams[0]
    I = beamparams[1]
    rho = beamparams[2]
    A = beamparams[3]
    L = beamparams[4]
    npoints = 2001
    i = 0
    w = sp.linspace(fmin, fmax, 2001) * 2 * sp.pi
    if min([xin, xout]) < 0 or max([xin, xout]) > L:
        print('One or both locations are not on the beam')
        return
    wn = sp.array((0, 0))
    # The number 100 is arbitrarily large and unjustified.
    a = sp.empty([npoints, 100], dtype=complex)
    f = sp.empty(100)

    while wn[-1] < 1.3 * (fmax * 2 * sp.pi):
        i = i + 1
        wn, xx, U = euler_beam_modes(n=i, bctype=bctype,
                                     beamparams=beamparams, npoints=5000)
        spl = UnivariateSpline(xx, U[:, i - 1])
        Uin = spl(xin)
        Uout = spl(xout)
        a[:, i - 1] = rho * A * Uin * Uout / \
            (wn[-1] ** 2 - w ** 2 + 2 * zeta * wn[-1] * w * sp.sqrt(-1))
        f[i] = wn[-1] / 2 / sp.pi
    a = a[:, 0:i]
    plt.figure()
    plt.subplot(211)
    plt.plot(w / 2 / sp.pi, 20 * sp.log10(sp.absolute(sp.sum(a, axis=1))), '-')
    # plt.hold('on')
    plt.plot(w / 2 / sp.pi, 20 * sp.log10(sp.absolute(a)), '-')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FRF (dB)')
    axlim = plt.axis()

    plt.axis(axlim + sp.array([0, 0, -0.1 * (axlim[3] - axlim[2]),
                               0.1 * (axlim[3] - axlim[2])]))

    plt.subplot(212)
    plt.plot(w / 2 / sp.pi, sp.unwrap(sp.angle(sp.sum(a, axis=1))) /
             sp.pi * 180, '-')
    plt.plot(w / 2 / sp.pi, sp.unwrap(sp.angle(a)) / sp.pi * 180, '-')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    axlim = plt.axis()
    plt.axis(axlim + sp.array([0, 0, -0.1 * (axlim[3] - axlim[2]),
                               0.1 * (axlim[3] - axlim[2])]))
    plt.show()

    fout = w / 2 / sp.pi
    H = a
    return fout, H


def ebf(xin, xout, fmin, fmax, zeta):
    """Shortcut call to `euler_beam_frf`."""
    _, _ = euler_beam_frf(xin, xout, fmin, fmax, zeta)
    return


def ebf1(xin, xout):
    """Shortcut call to `euler_beam_frf`."""
    _, _ = euler_beam_frf(xin, xout)
    return


def uniform_bar_modes(n=10, bctype=3, npoints=2001,
                      barparams=np.array([7.31e10, 2747.0, 0.4]),
                      kl_over_EA = 1000, m_over_rhoAL = 1000):
    """Mode shapes and natural frequencies of Uniform bar/rod.

    Parameters
    ----------
    n: int, numpy array
        highest mode number or array of mode numbers to return
    bctype: int
        bctype = 1 free-free
        bctype = 2 fixed-free
        bctype = 3 fixed-fixed
        bctype = 4 fixed-spring
        bctype = 5 fixed-mass
    barparams: numpy array
        E, rho, L
        Young's modulus, density, length of bar
    npoints: int
        number of points for returned mode shape array
    kl_over_EA: float
        spring stiffness times rod length divided by EA. (for clamped-spring
        condition)
    m_over_rhoAL: float
        end mass divided by rho A L of rod. (for end mass condition)

    Returns
    -------
    omega_n: numpy array
        array of natural frequencies
    x: numpy array
        x coordinate
    U: numpy array
        mass normalized mode shape

    Notes
    -----
    For most situations the cross sectional area cancels out and has no effect. It only matters in
    the last two cases: and end spring or and mass, and is included with the parameters special to
    those cases.



    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb
    >>> omega_n, x, U = vtb.uniform_bar_modes(n=3)
    >>> plt.figure()
    <matplotlib.figure...>
    >>> plt.plot(x,U)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('x (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('Displacement (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.title('Mode 3')
    <matplotlib.text.Text object at ...>
    """
    E = barparams[0]
    rho = barparams[1]
    L = barparams[2]
    if isinstance(n, int):
        ln = n
        n = np.arange(n) + 1
    else:
        ln = len(n)

    # len=[0:(1/(npoints-1)):1]';  %Normalized length of the bar
    x_normed = np.linspace(0, 1, npoints, endpoint = True)
    x = x_normed * L
    # Determine natural frequencies and mode shapes depending on the
    # boundary condition.
    # Mass simplification. The following was arange_(1,length_(n)).reshape(-1)
    mode_num_range = np.arange(1, ln)
    w = np.empty(ln)
    U = np.empty([npoints, ln])

    if bctype == 1:
        desc = 'Free-Free '
        for i in mode_num_range:
            w[i] = i * np.pi * np.sqrt(E/rho) / L
            U[:, i] = np.cos(i * np.pi * x_normed)
    elif bctype == 2:
        desc = 'Fixed-Free '
        for i in mode_num_range:
            w[i] = (2*i-1) * np.pi * np.sqrt(E/rho) / (2 * L)
            U[:, i] = np.sin((2*i-1) * np.pi * x_normed / 2)
    elif bctype == 3:
        desc = 'Fixed-Fixed '
        for i in mode_num_range:
            w[i] = i * np.pi * np.sqrt(E/rho) / L
            U[:, i] = np.sin((i) * np.pi * x_normed)
    elif bctype == 4:
        desc = 'Fixed-Spring'
        def func(lam):
            return lam + np.tan(lam)* kl_over_EA
        for i in mode_num_range:
            lam = newton_krylov(func, (0.25+i/2)*np.pi)
            w[i] = lam * np.sqrt(E/rho) / L
            U[:, i] = np.sin(lam * x_normed)
    elif bctype == 5:
        desc = 'Fixed-Mass'
        def func(lam):
            return np.tan(lam)-1/lam/m_over_rhoAL
        for i in mode_num_range:
            lam = newton_krylov(func, 0.25+i*np.pi)
            w[i] = lam * np.sqrt(E/rho) / L
            U[:, i] = np.sin(lam * x_normed)



    # Mass Normalization of mode shapes
    for i in mode_num_range:
        U[:, i] = U[:, i] / np.sqrt(np.dot(U[:, i], U[:, i]) * rho * L)

    omega_n = w
    return omega_n, x, U

"""
def ebf(xin, xout, fmin, fmax, zeta):
    _, _ = uniform_bar_frf(xin, xout, fmin, fmax, zeta)
    return
"""

def uniform_bar_modes(n=10, bctype=3, npoints=2001,
                      barparams=np.array([7.31e10, 2747.0, 0.4])):
    """Mode shapes and natural frequencies of Uniform bar/rod.

    Parameters
    ----------
    n: int, numpy array
        highest mode number or array of mode numbers to return
    bctype: int
        bctype = 1 free-free
        bctype = 2 fixed-free
        bctype = 3 fixed-fixed

    barparams: numpy array
        E, rho, L
        Young's modulus, density, length of bar
    npoints: int
        number of points for returned mode shape array


    Returns
    -------
    omega_n: numpy array
        array of natural frequencies
    x: numpy array
        x coordinate
    U: numpy array
        mass normalized mode shape


    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb
    >>> omega_n, x, U, *_ = vtb.uniform_bar_modes(n=1)
    >>>
    >>> plt.plot(x,U)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('x (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('Displacement (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.title('Mode 1')
    <matplotlib.text.Text object at ...>
    >>> plt.show()
    """
    E = barparams[0]
    rho = barparams[1]
    L = barparams[2]
    if isinstance(n, int):
        ln = n
        n = np.arange(n) + 1
    else:
        ln = len(n)

    # len=[0:(1/(npoints-1)):1]';  %Normalized length of the bar
    lenth = np.linspace(0, L, npoints)
    x = lenth * L
    # Determine natural frequencies and mode shapes depending on the
    # boundary condition.
    # Mass simplification. The following was arange_(1,length_(n)).reshape(-1)
    mode_num_range = np.arange(0, ln)
    w = np.empty(ln)
    U = np.empty([npoints, ln])

    if bctype == 1:
        desc = 'Free-Free '
        for i in mode_num_range:
            w[i] = i * np.pi * np.sqrt(E / rho) / L
            U[:, i] = np.cos(i * np.pi * lenth / L)
    elif bctype == 2:
        desc = 'Fixed-Free '
        for i in mode_num_range:
            w[i] = (2 * i - 1) * np.pi * np.sqrt(E / rho) / (2 * L)
            U[:, i] = np.sin((2 * i - 1) * np.pi * lenth / (2 * L))
    elif bctype == 3:
        desc = 'Fixed-Fixed '
        for i in mode_num_range:
            w[i] = i * np.pi * np.sqrt(E / rho) / L
            U[:, i] = np.sin(i * np.pi * lenth / (2 * L))
            # Mass Normalization of mode shapes
            # for i in mode_num_range:
            # U[:, i] = U[:, i] / np.sqrt(np.dot(U[:, i], U[:, i]) * rho * L)

    omega_n = w
    return omega_n, x, U

def torsional_bar_modes(n=10, bctype=2, cstype=4, npoints=2001,
                     tbarparams=np.array([7.8e9, 8.4375e-09,
                                          2747.0, 0.4]), cspar=np.array([0.7,0.9,.1,.2])):
    """Mode shapes and natural frequencies of Torsional bar.

    Parameters
    ----------
    n: int, numpy array
        highest mode number or array of mode numbers to return
    bctype: int
        bctype = 1 free-free
        bctype = 2 fixed-free
        bctype = 3 fixed-fixed
    tbarparams: numpy array
        G, J, rho, L
        Shear modulus, Polar moment of area, density,
        length of bar
    npoints: int
        number of points for returned mode shape array
    cspar: float
        Cross-section parameters
    cstype: numpy array
        Cross-section type
        cstype =1 is a circular shaft and cspar = R.
        cstype = 2 is a hollow circular shaft and cspar = [R1 R2]
        cstype = 3 is a square shaft and cspar = a
        cstype = 4 is a hollow rectangular shaft and cspar = [a b A B]

    Returns
    -------
    omega_n: numpy array
        array of natural frequencies
    x: numpy array
        x coordinate
    U: numpy array
        mass normalized mode shape

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb
    >>> omega_n, x, U,*_ = vtb.torsional_bar_modes(n=1)
    >>>
    >>> plt.plot(x,U)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('x (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('Displacement (m)')
    <matplotlib.text.Text object at ...>
    >>> plt.title('Mode 1')
    <matplotlib.text.Text object at ...>
    >>> plt.grid('on')
    """

    G = tbarparams[0]
    J = tbarparams[1]
    rho = tbarparams[2]
    L = tbarparams[3]

    if cstype == 1:
        R = cspar[0]
        g = (np.pi * R ** 4) / 2
    elif cstype == 2:
        R1 = cspar[0]
        R2 = cspar[1]
        g = (np.pi / 2) * (R2 ** 4 - R1 ** 4)
    elif cstype == 3:
        a = cspar[1]
        g = .1406 * a ** 4
    elif cstype == 4:
        if len(cspar) != 4:
            print("Not enough parameters for cstype 4")
        a = cspar[0]
        b = cspar[1]
        A = cspar[2]
        B = cspar[3]
        g = (2 * A * B * (a - A) ** 2 * (b - B) ** 2) / (a * A + b * B - A ** 2 - B ** 2)

    if g<=0:
        print("The constant gamma is less than or equal to zero.")

    if isinstance(n, int):
        ln = n
        n = np.arange(n) + 1
    else:
        ln = len(n)

    lenth = np.linspace(0, 1, npoints)
    x = lenth * L

    # Determine natural frequencies and mode shapes depending on the
    # boundary condition.
    # Mass simplification. The following was arange_(1,length_(n)).reshape(-1)
    mode_num_range = np.arange(0, ln)
    w = np.empty(ln)
    U = np.empty([npoints, ln])
    c = np.sqrt(G*g/(rho*J))

    if bctype == 1:
        desc = 'Free-Free '
        for i in mode_num_range:
            w[i] = (i * np.pi * c) / L
            U[:, i] = np.cos(i * np.pi * x / L)
    elif bctype == 2:
        desc = 'Fixed-Free '
        for i in mode_num_range:
            w[i] = (2 * i - 1) * np.pi * c / (2 * L)
            U[:, i] = np.sin((2 * i - 1) * np.pi * x / (2 * L))
    elif bctype == 3:
        desc = 'Fixed-Fixed '
        for i in mode_num_range:
            w[i] = (i * np.pi * c) / L
            U[:, i] = np.sin(i * np.pi * x / L)

    # Mass Normalization of mode shapes
    for i in mode_num_range:
        U[:, i] = U[:, i] / np.sqrt(np.dot(U[:, i], U[:, i]) * rho * L)

    omega_n = w
    return omega_n, x, U

