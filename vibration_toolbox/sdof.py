"""Single degree of freedom responses and plots."""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import fft
import matplotlib as mpl
from scipy import integrate

try:
    from IPython.display import clear_output, display, HTML
    from ipywidgets import interact, interact_manual, FloatSlider
    from ipywidgets import interactive, HBox, VBox, widgets, Label
except ImportError:
    print('Interactive iPython tools will not work without IPython.display \
          and ipywidgets installed.')


def _in_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole?
            return True
        elif shell == 'TerminalInteractiveShell':  # Terminal running IPython?
            return False
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (10, 6)


def free_response(m=10, c=1, k=100, x0=1, v0=-1, max_time=10):
    r"""Free response of a second order linear oscillator.

    Returns t, x, v, zeta, omega, omega_d and A resulting from the
    free response of a second order linear ordinary differential
    equation defined by
    :math:`m\ddot{x} + c \dot{x} + k x = 0`
    given initial conditions :math:`x_0` and :math:`\dot{x}_0 = v_0` for
    :math:`0 < t < t_{max}`

    Parameters
    ----------
    m, c, k :  floats, optional
        mass, damping coefficient, stiffness
    x0, v0:  floats, optional
        initial displacement, initial velocity
    max_time: float, optional
        end time for :math:`x(t)`

    Returns
    -------
    t, x, v : ndarrays
        time, displacement, and velocity
    zeta, omega, omega_d, A : floats
        damping ratio, undamped natural frequency, damped natural frequency,
        Amplitude

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb

    >>> t, x, *_ = vtb.free_response() # *_ ignores all other returns
    >>> plt.plot(t,x)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.xlabel('Time (sec)')
    Text(0.5, 0, 'Time (sec)')
    >>> plt.ylabel('Displacement (m)')
    Text(0, 0.5, 'Displacement (m)')
    >>> plt.title('Displacement versus time')
    Text(0.5, 1.0, 'Displacement versus time')
    >>> plt.grid(True)

    """
    omega = np.sqrt(k / m)
    zeta = c / 2 / omega / m
    omega_d = omega * np.sqrt(1 - zeta ** 2)
    A = np.sqrt(x0 ** 2 + (v0 + omega * zeta * x0) ** 2 / omega_d ** 2)

    def sdofs_deriv(x_xd, t0, m=m, c=c, k=k):
        x, xd = x_xd
        return [xd, -c / m * xd - k / m * x]

    z0 = np.array([[x0, v0]])
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250 * max_time))
    z_t = np.asarray([integrate.odeint(sdofs_deriv, z0i, t)
                      for z0i in z0])

    x, y = z_t[:, :].T
    return t, x, y, zeta, omega, omega_d, A


def phase_plot(m=10, c=1, k=100, x0=1, v0=-1, max_time=10):
    """Phase plot of free response of single degree of freedom system.

    For information on variables see `free_response`.

    Parameters
    ----------
    m, c, k:  floats, optional
        mass, damping coefficient, stiffness
    x0, v0:  floats, optional
        initial displacement, initial velocity
    max_time: float, optional
        end time for :math:`x(t)`

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibration_toolbox as vtb
    >>> vtb.phase_plot()

    """
    t, x, v, zeta, omega, omega_d, A = free_response(
        m, c, k, x0, v0, max_time)
    fig = plt.figure()
    fig.suptitle('Velocity vs Displacement')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Velocity')
    ax.grid(True)
    ax.plot(x, v)
    plt.show()


def phase_plot_i(max_time=(1.0, 200.0), v0=(-100, 100, 1.0),
                 m=(1.0, 100.0, 1.0),
                 c=(0.0, 1.0, 0.1), x0=(-100, 100, 1), k=(1.0, 100.0, 1.0)):
    """Interactive phase plot of free response of SDOF system.

    ``phase_plot_i`` is only functional in a
    `Jupyter notebook <http://jupyter.org>`_.

    Parameters
    ----------
    m, c, k:  floats, optional
        mass, damping coefficient, stiffness
    x0, v0:  floats, optional
        initial displacement, initial velocity
    max_time: float, optional
        end time for :math:`x(t)`

    """
    if _in_ipynb():
        w = interactive(phase_plot, max_time=max_time, v0=v0, m=m,
                        c=c, x0=x0, k=k)
        plt.show()
        display(w)
    else:
        print('phase_plot_i can only be used in an iPython notebook.')


def time_plot(m=10, c=1, k=100, x0=1, v0=-1, max_time=100):
    t, x, v, zeta, omega, omega_d, A = free_response(
        m, c, k, x0, v0, max_time)
    fig = plt.figure()
    fig.suptitle('Displacement vs Time')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.grid(True)
    ax.plot(t, x)
    if zeta < 1:
        ax.plot(t, A * np.exp(-zeta * omega * t), '--g',
                linewidth=1)
        ax.plot(t, -A * np.exp(-zeta * omega * t), '--g',
                linewidth=1, label=r'$A e^{- \zeta \omega t}$')
        tmin, tmax, xmin, xmax = ax.axis()
        ax.text(.75 * tmax, .85 * (xmax - xmin) + xmin,
                r'$\omega$ = %0.2f rad/sec' % (omega))
        ax.text(.75 * tmax, .80 * (xmax - xmin)
                + xmin, r'$\zeta$ = %0.2f' % (zeta))
        ax.text(.75 * tmax, .75 * (xmax - xmin) + xmin,
                r'$\omega_d$ = %0.2f rad/sec' % (omega_d))
    else:
        tmin, tmax, xmin, xmax = ax.axis()
        ax.text(.75 * tmax, .85 * (xmax - xmin)
                + xmin, r'$\zeta$ = %0.2f' % (zeta))
        ax.text(.75 * tmax, .80 * (xmax - xmin) + xmin,
                r'$\lambda_1$ = %0.2f' %
                (zeta * omega - omega * (zeta ** 2 - 1)))
        ax.text(.75 * tmax, .75 * (xmax - xmin) + xmin,
                r'$\lambda_2$ = %0.2f' %
                (zeta * omega + omega * (zeta ** 2 - 1)))
    ax.legend()
    plt.show()


def time_plot_i(max_time=(1.0, 100.0), x0=(-100, 100), v0=(-100, 100),
                m=(1.0, 100.0), c=(0.0, 1.0, .02), k=(1.0, 100.0)):
    """Interactive single degree of freedom free reponse plot in iPython.

    ``time_plot_i`` is only functional in a
    `Jupyter notebook <http://jupyter.org>`_.

    Parameters
    ----------
    m, c, k:  floats, optional
        mass, damping coefficient, stiffness
    x0, v0:  floats, optional
        initial displacement, initial velocity
    max_time: float, optional
        end time for :math:`x(t)`

    """
    if _in_ipynb():
        w = interactive(time_plot, max_time=max_time, v0=v0, m=m,
                        c=c, x0=x0, k=k)
        display(w)
    else:
        print('time_plot_i can only be used in an iPython notebook.')
    '''
    I'd like to get the sliders to be side by side to take less vertical
    space
    # cont = widgets.HBox(children = w)
    print(help(w))
    '''


def analytical(m=1, c=0.1, k=1, x0=1, v0=0, n=8, dt=0.05):
    """Return x(t) of analytical solution."""
    w = np.sqrt(k / m)
    zeta = c / (2 * w * m)  # (1.30)

    wd = w * np.sqrt(1 - zeta**2)  # (1.37)
    t = np.linspace(0, n * dt, n + 1)

    print('The natural frequency is ', w, 'rad/s.')
    print('The damping ratio is ', zeta)
    print('The damped natural frequency is ', wd)

    if zeta < 1:
        A = np.sqrt(((v0 + zeta * w * x0)**2 + (x0 * wd)**2) / wd**2)  # (1.38)
        phi = np.arctan2(x0 * wd, v0 + zeta * w * x0)  # (1.38)
        x = A * np.exp(-zeta * w * t) * np.sin(wd * t + phi)  # (1.36)
        print('A =', A)
        print('phi =', phi)

    elif zeta == 1:
        a1 = x0  # (1.46)
        a2 = v0 + w * x0  # (1.46)
        print('a1= ', a1)
        print('a2= ', a2)
        x = (a1 + a2 * t) * np.exp(-w * t)  # (1.45)

    else:
        a1 = (-v0 + (-zeta + np.sqrt(zeta**2 - 1)) * w * x0) / \
            (2 * w * np.sqrt(zeta**2 - 1))  # (1.42)
        a2 = (v0 + (zeta + np.sqrt(zeta**2 - 1)) * w * x0) / \
            (2 * w * np.sqrt(zeta**2 - 1))  # (1.43)
        print('a1= ', a1)
        print('a2= ', a2)
        x = (np.exp(-zeta * w * t)
             * (a1 * np.exp(-w * np.sqrt(zeta**2 - 1) * t)
             + a2 * np.exp(w * np.sqrt(zeta**2 - 1) * t)))  # (1.41)
    return x


def _euler(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05):
    """Deprecated Euler method free response of a SDOF system (program demo).

    Free response using Euler's method to perform numerical integration.

    Parameters
    ----------
    m, c, k: float
        Mass, damping and stiffness.
    x0, v0: float
        Initial conditions
    n: int
        The number of steps
    dt: float
        The step size.

    Returns
    -------
    t, x, v: array
        Time, displacement, and velocity

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> vtb._euler(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05)  # doctest: +SKIP
    (array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 ]),
    array([[ 1.  ,  0.  ],
           [ 1.  , -0.05],
           [ 1.  , -0.1 ],
           [ 0.99, -0.15],
           [ 0.99, -0.2 ],
           [ 0.98, -0.25],
           [ 0.96, -0.29],
           [ 0.95, -0.34],
           [ 0.93, -0.39]]))
    """
    # creates the state space matrix
    A = np.array([[0, 1],
                  [-k / m, -c / m]])
    # creates the x array and set the first line according to the initial
    # conditions
    x = np.zeros((n + 1, 2))
    x[0] = x0, v0

    for i in range(0, n):
        x[i + 1] = x[i] + dt * A@x[i]

    t = np.linspace(0, n * dt, n + 1)

    return t, x


def _rk4(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05):
    """Runge-Kutta solution of underdamped system.

    Parameters
    ----------
    m, c, k: float
        Mass, damping and stiffness.
    x0, v0: float
        Initial conditions
    n: int
        The number of steps
    dt: float
        The step size.

    Returns
    -------
    t, x, v: array
        Time, displacement, and velocity

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> vtb._rk4(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05) # doctest: +SKIP
    (array([0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 ]),
    array([[ 1.,  0.],
           [ 1.  , -0.05],
           [ 1.  , -0.1 ],
           [ 0.99, -0.15],
           [ 0.98, -0.2 ],
           [ 0.97, -0.24],
           [ 0.96, -0.29],
           [ 0.94, -0.34],
           [ 0.92, -0.38]]))

    """
    t = np.linspace(0, n * dt, n + 1)
    x = np.zeros((n + 1, 2))
    x[0, :] = x0, v0
    A = np.array([[0, 1],
                  [-k / m, -c / m]])

    def f(x_): return A@x_

    for i in range(n):
        k1 = dt * f(x[i])
        k2 = dt * f(x[i] + k1 / 2)
        k3 = dt * f(x[i] + k2 / 2)
        k4 = dt * f(x[i] + k3)
        x[i + 1] = x[i] + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    return t, x


# def
# euler_beam_frf(xin=0.22,xout=0.22,fmin=0.0,fmax=1000.0,beamparams=np.array((7.31e10,
# 1/12*0.03*.015**3, 2747, .015*0.03, 0.4)),


def frfplot(f, H):
    """Simply plot frequency response function from FRF data.

    Plots, scales, and labels Frequency Response Function plots.
    This simply saves a few common tasks when plotting data.

    Parameters
    ----------
    f: float array
    H: complex float array

    See Also
    --------
    vibrationtesting.frfplot : Plots FRF in a variety of formats

    """
    plt.subplot(211)
    plt.plot(f, 20 * np.log10(np.absolute(np.sum(H, axis=1))), '-')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FRF (dB)')
    axlim = plt.axis()
    plt.axis(axlim + np.array([0, 0, -0.1 * (axlim[3] - axlim[2]),
                               0.1 * (axlim[3] - axlim[2])]))

    plt.subplot(212)
    plt.plot(f, np.unwrap(np.angle(np.sum(H, axis=1))) / np.pi * 180, '-')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    axlim = plt.axis()
    plt.axis(axlim + np.array([0, 0, -0.1 * (axlim[3] - axlim[2]),
                               0.1 * (axlim[3] - axlim[2])]))


def response(xdd, f, t, x0, v0):
    r"""returns t, x, v- incomplete.

    :math:`\ddot{x} = g(x,v) + f(t)`
    given initial conditions :math:`x_0` and :math:`\dot{x}_0 = v_0` for the time `t`

*** Function hasn't been written yet... work in progress. Mimics vtb1_3
    Parameters
    ----------
    m, c, k:           1) Floats. Mass, damping and stiffness.
    x0, v0:            2) Floats. Initial conditions
    max_time:          3) Float. end time or response to be returned

    Returns
    -------
    t, x, v: 1) Arrays. Time, displacement, and velocity

    Examples
    --------
    >>> import vibration_toolbox as vtb


    """
    omega = np.sqrt(k / m)
    zeta = c / 2 / omega / m
    omega_d = omega * np.sqrt(1 - zeta**2)
    A = np.sqrt(x0**2 + (v0 + omega * zeta * x0)**2 / omega_d**2)
#    print('The natural frequency is ', omega, 'rad/s.');
#    print('The damping ratio is ', zeta);
#    print('The damped natural frequency is ', omega_d);

    def sdofs_deriv(x_xd, t0, m=m, c=c, k=k):
        x, xd = x_xd
        return [xd, -c / m * xd - k / m * x]

    z0 = np.array([[x0, v0]])
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250 * max_time))
    z_t = np.asarray([integrate.odeint(sdofs_deriv, z0i, t)
                      for z0i in z0])

    x, y = z_t[:, :].T
    return t, x, y, zeta, omega, omega_d, A


def _forced_analytical(m=10, k=100, x0=1, v0=0,
                       wdr=0.5, F0=10, tf=100):
    """Deprecated Return response of an undamped SDOFS to sinusiod.

    Parameters
    ----------
    m, k: float
        Mass and stiffness
    x0, v0: float
        Initial conditions
    wdr:
        Force frequency
    F0: float
        Force magnitude
    tf: float
        End time

    Returns
    -------
    t, x: array
        Time and displacement


    """
    # >>> _forced_analytical(m=10, k=100, x0=1, v0=0, wdr=0.5, F0=10, tf=100)

    t = np.linspace(0, tf, tf / 0.000125)

    f0 = F0 / m
    w = np.sqrt(k / m)
    x = (v0 / w * np.sin(w * t)
         + (x0 - f0 / (w**2 - wdr**2)) * np.cos(w * t)
         + f0 / (w**2 - wdr**2) * np.cos(wdr * t))   # (2.11)

    return t, x


def forced_response(m=10, c=0, k=100, x0=1, v0=0,
                    wdr=0.5, F0=10, max_time=100):
    r"""Harmonic response of SDOF system.

    Returns the the response of an underdamped single degree of
    freedom system to a sinusoidal input with amplitude F0 and
    frequency :math:`\omega_{dr}`.

    Parameters
    ----------
    m, c, k: float, optional
        Mass Damping, and stiffness
    x0, v0: float, optional
        Initial conditions
    wdr: float, optional
        Force frequency
    F0: float, optional
        Force magnitude
    max_time: float, optional
        End time

    Returns
    -------
    t, x, v: array
        Time, displacement and velocity

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> t, x, v = vtb.forced_response(m=10, c=0, k=100, x0=1, v0=0, wdr=0.5, F0=10, max_time=100)
    >>> plt.plot(t,x,t,v)
    [<matplotlib.lines.Line2D ...

    """
    def sdofs_deriv(x_xd, t, m=m, c=c, k=k):
        x, xd = x_xd
        return [xd, (F0 * np.cos(wdr * t) / m) - (c / m) * xd - (k / m) * x]

    z0 = np.array([x0, v0])
    # Solve for the trajectories
    t = np.linspace(0, max_time, int(250 * max_time))
    z_t = integrate.odeint(sdofs_deriv, z0, t)

    x, y = z_t.T
    return t, x, y


def steady_state_response(zs=0.1, rmin=0.0, rmax=2.0):
    """Plot steady state response of SDOF damped system.

    Parameters
    ----------
    zs: array
        Array with the damping values
    rmin, rmax: floats
        Minimum and maximum frequency ratio

    Returns
    -------
    r: Array
        Array containing the values for the frequency ratio
    A: Array
        Array containing the values for anmplitude

        Plot with steady state magnitude and phase

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> r, A = vtb.steady_state_response([0.1, 0.3, 0.8], 0, 2)

    """
    if not isinstance(zs, list):
        zs = [zs]
    r = np.linspace(rmin, rmax, int(100 * (rmax - rmin)))
    A0 = np.zeros((len(zs), len(r)), complex)
    for z in enumerate(zs):
        A0[z[0]] = (1 / (1 - r**2 + 2 * 1j * r * z[1]))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    plt.tight_layout()

    ax1.set_ylabel('Normalized Amplitude (dB)')
    ax1.set_title('Normalized Amplitude vs Frequency Ratio')

    ax2.set_xlabel('Frequency Ratio')
    ax2.set_ylabel('Phase Lag (deg)')
    ax2.set_title('Phase vs Frequency Ratio')

    for A in A0:
        ax1.plot(r, (np.absolute(A)))
        ax2.plot(r, -np.angle(A) / np.pi * 180)

    ax1.legend(([r'$\zeta$ = ' + (str(s)) for s in zs]))
    plt.show()
    return r, A


def steady_state_response_i(zs=(0, 1.0, 0.1), rmin=(0, 1, .1),
                            rmax=(1., 2.0, 0.1)):
    """Interactive phase plot of SDOF system.

    ``steady_state_response`` is only functional in a
    `Jupyter notebook <http://jupyter.org>`_.


    Parameters
    ----------
    zs: array
        Array with the damping values
    rmin, rmax: floats
        Minimum and maximum frequency ratio

    Returns
    -------
    r: Array
        Array containing the values for the frequency ratio
    A: Array
        Array containing the values for anmplitude

        Plot with steady state magnitude and phase

    """
    if _in_ipynb():
        w = interactive(steady_state_response, zs=zs,
                        rmin=rmin, rmax=rmax)
        display(w)
    else:
        print('steady_state_response_i can only be used in an iPython\
              notebook.')


def transmissibility(zs=np.array([0.05, 0.1, 0.25, 0.5, -0.75]),
                     rmin=0.,
                     rmax=2.):
    """Plot transmissibility ratio for SDOF system.

    Parameters
    ----------
    zs: array
        Array with the damping values
    rmin, rmax: floats
        Minimum and maximum frequency ratios

    Returns
    -------
    r: float array
        Array containing the values for the frequency ratio
    D: float array
        Array containing the values for displacement
    F: float array
        Array containing the values for force

        Plot with Displacement transmissibility ratio
        and force transmissibility ratio

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> r, D, F = vtb.transmissibility(zs=[0.1, 0.2], rmin=0, rmax=2)

    """
    if not isinstance(zs, list):
        zs = [zs]
    r = np.linspace(rmin, rmax, int(300 * (rmax - rmin)))
    DT = np.zeros((len(zs), len(r)))
    for z in enumerate(zs):
        # 2.71
        DT[z[0]] = ((1 + (2 * z[1] * r)**2)
                    / ((1 - r**2)**2 + (2 * z[1] * r)**2))**0.5

    FT = (r**2) * DT

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    plt.tight_layout()

    ax1.set_ylabel('Displacement Transmissibility Ratio (dB)')
    ax1.set_title(
        'Displacement Transmissibility Ratio vs Frequency Ratio (X/Y)')
    # ax1.set_ylim(0, 6)

    ax2.set_xlabel('Frequency Ratio')
    ax2.set_ylabel('Force Transmissibility Ratio (dB)')
    ax2.set_title(
        'Force Transmissibility Ratio versus Frequency Ratio (F_T/kY)')
    ax2.set_yscale("log")
    # ax2.set_ylim(0.01, 50)

    for D in DT:
        ax1.plot(r, D)
    for F in FT:
        ax2.plot(r, F)

    ax1.legend(([r'$\zeta$ = ' + (str(s)) for s in zs]))
    plt.show()
    return r, D, F


def transmissibility_i(zs=(0, 1.0, 0.1), rmin=0, rmax=2.0):
    """Interactive transmissibility phase of SDOF system.

    ``transmissibility_i`` is only functional in a
    `Jupyter notebook <http://jupyter.org>`_.

    Parameters
    ----------
    zs: array
        Array with the damping values
    rmin, rmax: float
        Minimum and maximum frequency ratio

    Returns
    -------
    r: Array
        Array containing the values for the frequency ratio
    D: Array
        Array containing the values for displacement
    F: Array
        Array containing the values for force

        Plot with Displacement transmissibility ratio
        and force transmissibility ratio

    """
    if _in_ipynb():
        w = interactive(transmissibility, zs=zs,
                        rmin=rmin, rmax=rmax)
        display(w)
    else:
        print('transmissibility_i can only be used in an iPython notebook.')


def rotating_unbalance(m, m0, e, zs, rmin, rmax, normalized=True):
    """Plot displacement of system responding to rotating unbalance.

    Parameters
    ----------
    m: float
        Mass of the system
    m0, e: float
        Mass and eccentricity of the unbalance.
    zs: array
        Array with the damping values
    rmin, rmax: float
        Minimum and maximum frequency ratio
    normalized: bool
        If true, the displacement is normalized (m*X/(m0*e))

    Returns
    -------
    r: Array
        Array containing the values for the frequency ratio
    Xn: Array
        Array containing the values for displacement

        Plot with Displacement displacement and phase
        for a system with rotating unbalance.

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> r, Xn = vtb.rotating_unbalance(m=1, m0=0.5, e=0.1, zs=[0.1, 0.25, 0.707, 1], rmin=0, rmax=3.5, normalized=True)

    """
    if not isinstance(zs, list):
        zs = [zs]
    r = np.linspace(rmin, rmax, int(100 * (rmax - rmin)))
    Xn = np.zeros((len(zs), len(r)), complex)
    for z in enumerate(zs):
        Xn[z[0]] = (r / (1 - r**2 + 2 * 1j * r * z[1]))

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    plt.tight_layout()

    if normalized is False:
        Xn = Xn * (m0 * e / m)
        ax1.set_ylabel('Displacement Magnitude')
        ax1.set_title('Displacement Magnitude vs Frequency Ratio')
    else:
        ax1.set_ylabel('Normalized Displacement Magnitude')
        ax1.set_title('Normalized Displacement Magnitude vs Frequency Ratio')

    ax2.set_xlabel('Frequency Ratio')
    ax2.set_ylabel('Phase')
    ax2.set_title('Phase vs Frequency Ratio')

    for X_z in Xn:
        ax1.plot(r, np.absolute(X_z))
        ax2.plot(r, -np.angle(X_z) / np.pi * 180)

    ax1.legend(([r'$\zeta$ = ' + (str(s)) for s in zs]))
    plt.show()
    return r, Xn


def impulse_response(m, c, k, Fo, max_time):
    r"""Plot SDOF system impulse response.

    Returns a plot with the response of a SDOF system to an
    impulse of magnitude Fo (Newton seconds).

    :math:`x(t)=\frac{F_o}{m\omega_d}e^{-\zeta\omega_n_t}\sin(\omega_d t-\phi)`

    Parameters
    ----------
    m, c, k: float
        Mass, damping and stiffness.
    Fo: float
        Force applied over time (units N.s)
    max_time: float
        End time

    Returns
    -------
    t: Array
        Array containing the values for the time
    x: Array
        Array containing the values for displacement

        Plot with the response of the system to an
        impulse of magnitude Fo (N.s).

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> t, x = vtb.impulse_response(m=100, c=20, k=2000, Fo=10, max_time=50)

    """
    t = np.linspace(0, max_time, int(250 * max_time))

    wn = np.sqrt(k / m)
    zeta = c / (2 * wn * m)
    wd = wn * np.sqrt(1 - zeta**2)
    fo = Fo / m

    x = fo / (wd * np.exp(zeta * wn * t)) * np.sin(wd * t)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Displacement')
    ax1.set_title('Displacement vs Time')
    ax1.plot(t, x)

    return t, x


def step_response(m, c, k, Fo, max_time):
    r"""Plot of step response of SDOF system.

    :math:`x(t)=\frac{F_0}{k}(1-\frac{1}{\sqrt{1-\zeta^2}}e^{-\zeta\omega_nt}\cos(\omega_d t -\theta))`

    where :math:`\theta = atan\frac{\zeta}{1-\zeta^2}`

    Parameters
    ----------
    m, c, k: float
        Mass, damping and stiffness.
    Fo: float
        Force applied
    max_time: float
        End time

    Returns
    -------
    t: Array
        Array containing the values for the time
    x: Array
        Array containing the values for displacement

        Plot with the response of the system to an
        step of magnitude Fo.

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> t, x = vtb.step_response(m=100, c=20, k=2000, Fo=10, max_time=100)

    """
    t = np.linspace(0, max_time, int(250 * max_time))

    wn = np.sqrt(k / m)
    zeta = c / (2 * wn * m)
    wd = wn * np.sqrt(1 - zeta**2)
    fo = Fo / m

    if zeta != 1:
        phi = np.arctan(zeta / np.sqrt(1 - zeta**2))

    if 0 < zeta < 1:
        x = fo / wn**2 * (1 - wn / wd * np.exp(-zeta * wn * t)
                          * np.cos(wd * t - phi))
    elif zeta == 1:
        lam = -wn
        A1 = -fo / wn**2
        A2 = -A1 * lam
        x = fo / wn**2 + A1 * np.exp(lam * t) + A2 * t * np.exp(lam * t)
    elif zeta > 1:
        lam1 = -zeta * wn - wn * np.sqrt(zeta**2 - 1)
        lam2 = -zeta * wn + wn * np.sqrt(zeta**2 - 1)
        A2 = fo / (wn**2 * (lam2 / lam1 - 1))
        A1 = -lam2 / lam1 * A2
        x = fo / wn**2 + A1 * np.exp(lam1 * t) + A2 * np.exp(lam2 * t)
    else:
        raise ValueError('Zeta should be greater than zero')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Displacement')
    ax1.set_title('Displacement vs Time')
    ax1.plot(t, x)

    return t, x


def fourier_series(dat, t, n):
    """Fourier series approximation to a function.

    returns Fourier coefficients of a function.
    The coefficients are numerical approximations of the true
    coefficients.

    Parameters
    ----------
    dat: array
        Array of data representing the function.
    t: array
        Corresponding time array.
    n: int
        The desired number of terms to use in the
        Fourier series.

    Returns
    -------
    a, b: tuple
        Tuple containing arrays with the Fourier coefficients
        as float arrays.
        The function also produces a plot of the approximation.

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> f = np.hstack((np.arange(-1, 1, .04), np.arange(1, -1, -.04)))
    >>> f += 1
    >>> t = np.arange(0, len(f))/len(f)
    >>> a, b = vtb.fourier_series(f, t, 5)

    """
    len_ = len(dat) / 2
    fs = (fft(dat)) / len_
    a0 = fs[0]
    a = np.real(np.hstack((a0, fs[1:len(fs / 2)])))
    b = -np.imag(fs[1:len(fs / 2)])
    len_ *= 2
    dt = 2 * np.pi / len_
    tp = np.arange(0, 2 * np.pi, dt)
    dataapprox = a[0] / 2 + np.zeros_like(dat)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(t, dat)

    for i in range(1, n):
        newdat = a[i] * np.cos(tp * i) + b[i] * np.sin(tp * i)
        dataapprox += newdat
        if i == n - 1:
            ax1.plot(t, newdat)

    ax1.plot(t, dataapprox)

    return a, b


def response_spectrum(f):
    """Plot response spectrum of ramp response.

    See Figure 3.13- Inman for the system with natural frequency
    f (in Hz) and no damping.

    Parameters
    ----------
    f: float
        Natural frequency.

    Returns
    -------
    t, rs: tuple
        Tuple with time and response arrays. It also returns
        a plot with the response spectrum.

    Examples
    --------
    >>> import vibration_toolbox as vtb
    >>> t, rs = vtb.response_spectrum(10)

    """
    t = np.linspace(.001 * 4 / f, 10 / f, 200)
    w = 2 * np.pi * f

    one = np.ones_like(t)

    rs1 = one / (w * t)
    rs2 = np.sqrt(2 * (1 - np.cos(w * t)))

    rs = one + rs1 * rs2

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Rise time (t_1)')
    ax1.set_ylabel('Dimensionless maximum response - (xk/Fo)max')
    ax1.set_title('Response spectrum of a SDOF system with f = %s Hz' % f)
    ax1.plot(t, rs)

    return t, rs


def fourier_approximation(a0, aodd, aeven, bodd, beven, N, T):
    r"""Plot the Fourier series defined by coefficient falues.

    Coefficients are defined by Inman [1]_.

    :math:`a_0=\frac{2}{T}\int_0^T F(t)dt`

    :math:`a_n=\frac{2}{T}\int_0^T F(t) \cos(n\omega_T t)dt`

    :math:`b_n=\frac{2}{T}\int_0^T F(t) \sin(n\omega_T t)dt`

    Parameters
    ----------
    a0: float or function
        :math:`a_0`-  Fourier coefficient.
    aodd: float or function
        :math:`a_n`-  Fourier coefficient for n odd.
    aeven: float or function
        :math:`a_n`-  Fourier coefficient for n even.
    bodd: float or function
        :math:`b_n`-  Fourier coefficient for n odd
    beven: float or function
        :math:`b_n`-  Fourier coefficient for n even

    Returns
    -------
    t, F: tuple
        Tuple with time and F(t). It also returns
        a plot with the Fourier approximation.

    References
    ----------
    .. [1] Daniel J. Inman, "Engineering Vibration", 4th ed., Prentice Hall,
           2013.

    Examples
    --------
    >>> # Square wave
    >>> import vibration_toolbox as vtb
    >>> bodd_square = lambda n: -3*(-1+(-1)**n)/n/np.pi
    >>> beven_square = lambda n: -3*(-1+(-1)**n)/n/np.pi
    >>> t, F = vtb.fourier_approximation(-1, 0, 0, bodd_square, beven_square, 20, 2)
    >>> # Triangular wave
    >>> aeven_triangle = lambda n: -8/np.pi**2/n**2
    >>> t, F = vtb.fourier_approximation(0,aeven_triangle,0,0,0,20,10)



    """
    def make_const(value):
        return lambda t: value

    def fourier_contribution(a, b, n):
        return a(n) * np.cos(n * 2 * np.pi * t / T) + \
               b(n) * np.sin(n * 2 * np.pi * t / T)

    a0, aodd, aeven, bodd, beven = (arg if callable(arg)
                                    else make_const(arg)
                                    for arg in (a0, aodd, aeven, bodd, beven))

    dt = min(T / 400, T / 10 * N)
    t = np.arange(0, T * 3, dt)
    F = 0 * t + a0(0) / 2

    # n odd
    for n in range(1, N, 2):
        F = F + fourier_contribution(aodd, bodd, n)
    # n even
    for n in range(2, N, 2):
        F = F + fourier_contribution(aeven, beven, n)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Time, t')
    ax1.set_ylabel('F(t)')
    ax1.plot(t, F)
    return t, F


def plot_sdof_resp(m=1.0, c=0.2, k=100.0):
    """Plot characteristic responses of SDOF system.

    Plot impluse response, step response, frequency response function, and
    location of poles.

    Parameters
    ----------
    m: float
        Mass
    c: float
        Damping
    k: float
        Stiffness

    """
    # This only does anything when the user accidentally sends in integers
    # instead of floats.
    if k == 0:
        k = 1e-10

    if m == 0:
        m = 1e-10

    omega_n = np.sqrt(k / m)

    zeta = c / 2 / np.sqrt(m * k)

    if zeta < 1.0 and np.abs(zeta) > 1e-6:
        omega_d = omega_n * np.sqrt(1 - zeta**2)

        # Approximately 4 times the settling time
        tmax = 4 / zeta / omega_n

        if tmax < 0:
            tmax = -tmax

        # guess reasonable amount of time - lock to first decimal place
        maxtime = (np.ceil(tmax / 10**np.floor(np.log10(tmax)))
                   * 10**np.floor(np.log10(tmax)))

        maxtime = 20.0

        num_cycles = maxtime * omega_n / 2 / np.pi

        num_points = np.max([200, int(10 * num_cycles)])

        if num_points > 1000:
            num_points = 1000
            print("""You don't have enough resolution to see
                     this plot. Expect aliasing issues.""")

        t = np.linspace(0, maxtime, num_points)

        decay = 1 / np.sqrt(1 - zeta**2) * np.exp(-zeta * omega_n * t)

        impulse_response = 1 / m * decay * np.sin(omega_d * t)

        phi = np.arctan(zeta / (np.sqrt(1 - zeta**2)))

        step_response = 1 / k * (1 - decay * np.cos(omega_d * t - phi))

        max_freq = omega_n * 4

        max_freq = (np.ceil(max_freq / 10**np.floor(np.log10(max_freq)))
                    * 10**np.floor(np.log10(max_freq)))

        omega = np.linspace(0, max_freq, num_points)
        s = sp.sqrt(-1.0) * omega
        freq_response = 1 / (m * s**2 + c * s + k)

        roots = np.array([[-zeta * omega_n - omega_d * sp.sqrt(-1.0)],
                          [-zeta * omega_n + omega_d * sp.sqrt(-1)]])

    elif zeta > 1.0:

        if zeta - 1 < 1e-8:
            print('adjusting zeta.')
            zeta = 1 + 1e-8

        roots = np.array([[(-c + np.sqrt(c**2 - 4 * m * k)) / 2 / c],
                          [(-c - np.sqrt(c**2 - 4 * m * k)) / 2 / c]])

        tmax = -4 / ((-c + np.sqrt(c**2 - 4 * m * k)) / 2 / c)
        # guess reasonable amount of time - lock to first decimal place
        maxtime = (np.ceil(tmax / 10**np.floor(np.log10(tmax)))
                   * 10**np.floor(np.log10(tmax)))

        maxtime = 20.0

        num_points = 300
        t = np.linspace(0, maxtime, num_points)

        # Impulse response

        x0 = 0
        v0 = 1 / m
        C1 = (x0 * omega_n * (zeta + np.sqrt(zeta**2 - 1)) + v0
              ) / 2 / omega_n / np.sqrt(zeta**2 - 1)
        C2 = (-x0 * omega_n * (zeta - np.sqrt(zeta**2 - 1)) - v0
              ) / 2 / omega_n / np.sqrt(zeta**2 - 1)
        impulse_response = C1 * np.exp(
            (-zeta + np.sqrt(zeta**2 - 1)) * omega_n * t) + C2 * np.exp(
                (-zeta - np.sqrt(zeta**2 - 1)) * omega_n * t)

        # Step response

        x0 = -1 / k
        v0 = 0 / m
        C1 = (x0 * omega_n * (zeta + np.sqrt(zeta**2 - 1)) + v0
              ) / 2 / omega_n / np.sqrt(zeta**2 - 1)
        C2 = (-x0 * omega_n * (zeta - np.sqrt(zeta**2 - 1)) - v0
              ) / 2 / omega_n / np.sqrt(zeta**2 - 1)
        step_response = 1 / k + C1 * np.exp(
            (-zeta + np.sqrt(zeta**2 - 1)) * omega_n * t) + C2 * np.exp(
                (-zeta - np.sqrt(zeta**2 - 1)) * omega_n * t)

        max_freq = omega_n * 4

        max_freq = (np.ceil(max_freq / 10**np.floor(np.log10(max_freq)))
                    * 10**np.floor(np.log10(max_freq)))

        omega = np.linspace(0, max_freq, num_points)
        s = sp.sqrt(-1.0) * omega
        freq_response = 1 / (m * s**2 + c * s + k)

    elif np.abs(zeta) < 1e-5:
        omega_d = omega_n

        # Approximately 8 cycles
        tmax = 8 * 2 * np.pi / omega_n

        # guess reasonable amount of time - lock to first decimal place
        maxtime = (np.ceil(tmax / 10**np.floor(np.log10(tmax)))
                   * 10**np.floor(np.log10(tmax)))

        num_cycles = maxtime * omega_n / 2 / np.pi

        num_points = np.max([200, int(10 * num_cycles)])

        if num_points > 1000:
            num_points = 1000
            print("""You don't have enough resolution to see this plot.
            Expect aliasing issues.""")

        t = np.linspace(0, maxtime, num_points)

        decay = np.zeros_like(t)

        impulse_response = 1 / m * np.sin(omega_d * t)

        step_response = 1 / k * (1 - np.cos(omega_d * t))

        max_freq = omega_n * 4

        max_freq = (np.ceil(max_freq / 10**np.floor(np.log10(max_freq)))
                    * 10**np.floor(np.log10(max_freq)))

        omega = np.linspace(0, max_freq, num_points)
        s = sp.sqrt(-1.0) * omega
        freq_response = 1 / (m * s**2 + c * s + k)

        roots = np.array([[-zeta * omega_n - omega_d * sp.sqrt(-1.0)],
                          [-zeta * omega_n + omega_d * sp.sqrt(-1)]])

    fig = plt.figure(figsize=(18, 10), dpi=80,
                     facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(221)
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$x(t)$')
    ax1.set_title('Impulse Response')
    ax1.plot(t, impulse_response)
    ax1.axis((0.0, 20.0, -0.3, 0.3))

    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('$t$')
    ax2.set_ylabel('$x(t)$')
    ax2.set_title('Step Response')
    ax2.plot(t, step_response)
    ax2.axis((0.0, 20.0, 0, 0.04))

    ax3 = fig.add_subplot(425)
    # ax3.set_xlabel('$\\omega$')
    ax3.set_ylabel('$20*log_{10}(|H(j\\omega)|)$')
    ax3.set_title('Frequency Response Magnitude')
    ax3.plot(omega, 20 * np.log10(np.abs(freq_response)))
    ax3.axis((0.0, 20.0, -80, -10))

    ax5 = fig.add_subplot(427)
    ax5.set_xlabel('$\\omega$')
    ax5.set_ylabel('$\\phi$')
    ax5.set_title('Frequency Response Phase')
    ax5.plot(omega, np.angle(freq_response) * 180 / np.pi)
    ax5.axis((0.0, 20.0, -200, 1))

    ax4 = fig.add_subplot(224)
    ax4.set_xlabel('Real')
    ax4.set_ylabel('Imaginary')
    ax4.set_title('Poles')
    ax4.axis('equal')
    # ax4.axis([-3, 3, -3, 3])
    ax4.axhline(y=0, color='k')
    ax4.axvline(x=0, color='k')
    ax4.plot(np.real(roots), np.imag(roots), '*')
    ax4.axis((-6.0, 6.0, -20.0, 20.0))
    plt.show()

    print('zeta = {:.3f}, omega_n = {:.2f}'.format(zeta, omega_n))


def sdof_interact():
    """Create interactive plots of characteristics of SDOF system."""
    m_slide = widgets.FloatSlider(min=1e-5, max=20, step=.1, value=5,
                                  continuous_update=False)
    k_slide = FloatSlider(min=1e-5, max=1000, step=1., value=100,
                          continuous_update=False)
    c_slide = FloatSlider(min=-1, max=200, step=.1, value=2,
                          continuous_update=False)

    m_label = widgets.Label('Mass')
    c_label = Label('Damping')
    k_label = Label('Stiffness')

    m_slider = widgets.VBox([m_label, m_slide])
    c_slider = widgets.VBox([c_label, c_slide])
    k_slider = widgets.VBox([k_label, k_slide])

    ui = widgets.HBox([m_slider, c_slider, k_slider])

    out = widgets.interactive_output(plot_sdof_resp,
                                     {'m': m_slide,
                                      'c': c_slide, 'k': k_slide})

    sdof_responses = widgets.VBox([ui, out])
    return sdof_responses
