import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
from ipywidgets import interact, interactive, widgets
from IPython.display import clear_output, display, HTML
from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (10, 6)
import IPython.core.display as ipcd


# from ipywidgets.widgets.interaction import interact, interactive

def solve_sdofs(m=10, c=1, k=100, x0=1, v0=-1, max_time=10):
    """
    returns t, x, v, zeta, omega, omega_d, A
    $\alpha$
    Returns free response of a second order linear ordinary differential equation
    defined by
    :math:`m\ddot{x} + c \dot{x} + k x = 0`
    given initial conditions :math:`x_0` and :math:`\dot{x}_0 = v_0` for
    :math:`0 < t < t_{max}`

    Parameters

    m, c, k:           1) Floats. Mass, damping and stiffness.
    x0, v0:            2) Floats. Initial conditions
    max_time:          3) Float. end time or response to be returned

    Returns

    t, x, v: 1) Arrays. Time, displacement, and velocity

    :Example:
    >>> solve_sdofs(m=10, c=1, k=100, x0=1, v0=-1, max_time=0.05)
    (array([ 0.        ,  0.00454545,  0.00909091,  0.01363636,  0.01818182,
            0.02272727,  0.02727273,  0.03181818,  0.03636364,  0.04090909,
            0.04545455,  0.05      ]), array([[ 1.        ],
           [ 0.99535245],
           [ 0.99050141],
           [ 0.98544797],
           [ 0.98019328],
           [ 0.9747385 ],
           [ 0.96908486],
           [ 0.96323361],
           [ 0.95718605],
           [ 0.95094353],
           [ 0.94450741],
           [ 0.93787913]]), array([[-1.        ],
           [-1.04488491],
           [-1.08953361],
           [-1.13393698],
           [-1.17808593],
           [-1.22197148],
           [-1.26558467],
           [-1.30891662],
           [-1.35195851],
           [-1.39470157],
           [-1.43713712],
           [-1.47925653]]), 0.015811388300841896, 3.1622776601683795, 3.1618823507524758, 1.0441611791969838)
    """

    omega = sp.sqrt(k / m)
    zeta = c / 2 / omega / m
    omega_d = omega * sp.sqrt(1 - zeta ** 2)
    A = sp.sqrt(x0 ** 2 + (v0 + omega * zeta * x0) ** 2 / omega_d ** 2)

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


def sdof_phase_plot(m=10, c=1, k=100, x0=1, v0=-1, max_time=10):
    '''Phase plot of free response of single degree of freedom system.
    For information on variables see `solve_sdofs`'''
    t, x, v, zeta, omega, omega_d, A = solve_sdofs(m, c, k, x0, v0, max_time)
    fig = plt.figure()
    fig.suptitle('Velocity vs Displacement')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Displacement')
    ax.set_ylabel('Velocity')
    ax.grid('on')
    ax.plot(x, v)


def sdof_phase_plot_i(max_time=(1.0, 200.0), v0=(-100, 100, 1.0), m=(1.0, 100.0, 1.0),
                      c=(0.0, 1.0, 0.1), x0=(-100, 100, 1), k=(1.0, 100.0, 1.0)):
    '''Interactive phase plot of free response of single degree of freedom system.
    For information on variables see ``solve_sdofs``'''
    w = interactive(sdof_phase_plot, max_time=max_time, v0=v0, m=m,
                    c=c, x0=x0, k=k)
    display(w)


def sdof_time_plot(m=10, c=1, k=100, x0=1, v0=-1, max_time=100):
    t, x, v, zeta, omega, omega_d, A = solve_sdofs(m, c, k, x0, v0, max_time)
    fig = plt.figure()
    fig.suptitle('Displacement vs Time')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.grid('on')
    ax.plot(t, x)
    if zeta < 1:
        ax.plot(t, A * sp.exp(-zeta * omega * t), '--', t, -A *
                sp.exp(-zeta * omega * t), '--g', linewidth=1)
        tmin, tmax, xmin, xmax = ax.axis()
        ax.text(.75 * tmax, .95 * (xmax - xmin) + xmin,
                '$\omega$ = %0.2f rad/sec' % (omega))
        ax.text(.75 * tmax, .90 * (xmax - xmin) +
                xmin, '$\zeta$ = %0.2f' % (zeta))
        ax.text(.75 * tmax, .85 * (xmax - xmin) + xmin,
                '$\omega_d$ = %0.2f rad/sec' % (omega_d))
    else:
        tmin, tmax, xmin, xmax = ax.axis()
        ax.text(.75 * tmax, .95 * (xmax - xmin) +
                xmin, '$\zeta$ = %0.2f' % (zeta))
        ax.text(.75 * tmax, .90 * (xmax - xmin) + xmin,
                '$\lambda_1$ = %0.2f' % (zeta * omega - omega * (zeta ** 2 - 1)))
        ax.text(.75 * tmax, .85 * (xmax - xmin) + xmin,
                '$\lambda_2$ = %0.2f' % (zeta * omega + omega * (zeta ** 2 - 1)))


def sdof_time_plot_i(max_time=(1.0, 100.0), v0=(-100, 100), m=(1.0, 100.0),
                     c=(0.0, 100.0), x0=(-100, 100), k=(1.0, 100.0)):
    w = interactive(sdof_time_plot, max_time=max_time, v0=v0, m=m,
                    c=c, x0=x0, k=k)
    # I'd like to get the sliders to be side by side to take less vertical space
    # cont = widgets.HBox(children = w)
    # print(help(w))
    display(w)


def sdof_analytical(m=1, c=0.1, k=1, x0=1, v0=0, tf=40):

    w = np.sqrt(k/m)
    zeta = c/(2*w*m)  # (1.30)

    wd = w*np.sqrt(1-zeta**2)  # (1.37)
    t = np.linspace(0, tf, 100000/tf)

    print('The natural frequency is ', w, 'rad/s.')
    print('The damping ratio is ', zeta)
    print('The damped natural frequency is ', wd)

    if zeta < 1:
        A = np.sqrt(((v0 + zeta*w*x0)**2 + (x0*wd)**2)/wd**2)  # (1.38)
        phi = np.arctan2(x0*wd, v0+zeta*w*x0)  # (1.38)
        x = A*np.exp(-zeta*w*t)*np.sin(wd*t+phi)  # (1.36)
        print('A =', A)
        print('phi =', phi)

    elif zeta == 1:
        a1 = x0  # (1.46)
        a2 = v0 + w*x0  # (1.46)
        print('a1= ', a1)
        print('a2= ', a2)
        x = (a1+a2*t)*np.exp(-w*t)  # (1.45)

    else:
        a1 = (-v0 + (-zeta + np.sqrt(zeta**2-1))*w*x0)/(2*w*np.sqrt(zeta**2-1))  # (1.42)
        a2 = (v0 + (zeta + np.sqrt(zeta**2-1))*w*x0)/(2*w*np.sqrt(zeta**2-1))  # (1.43)
        print('a1= ', a1)
        print('a2= ', a2)
        x = np.exp(-zeta*w*t)*(a1*np.exp(-w*np.sqrt(zeta**2-1)*t)+a2*np.exp(w*np.sqrt(zeta**2-1)*t))  # (1.41)

    return x


def sdof_euler(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05):
    """
    Returns free response of a second order linear ordinary differential equation
    using the Euler method for integration.

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
    ----------
    t, x, v: array
        Time, displacement, and velocity

    Examples:
    ----------
    >>> sdof_euler(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05)
    (array([ 0.  ,  0.05,  0.1 ,  0.15,  0.2 ,  0.25,  0.3 ,  0.35,  0.4 ]), array([[ 1.        ,  0.        ],
           [ 1.        , -0.05      ],
           [ 0.9975    , -0.09975   ],
           [ 0.9925125 , -0.14912625],
           [ 0.98505619, -0.19800624],
           [ 0.97515588, -0.24626902],
           [ 0.96284242, -0.29379547],
           [ 0.94815265, -0.34046861],
           [ 0.93112922, -0.3861739 ]]))

    """

    # creates the state matrix
    A = sp.array([[0, 1],
                  [-k / m, -c / m]])
    # creates the x array and set the first line according to the initial conditions
    x = sp.zeros((n + 1, 2))
    x[0] = x0, v0

    for i in range(0, n):
        x[i+1] = x[i] + dt*A@x[i]

    t = sp.linspace(0, n * dt, n + 1)

    return t, x


def sdof_rk4(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05):
    """
    Returns free response of a second order linear ordinary differential equation
    using the Runge-Kutta method for integration.

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
    ----------
    t, x, v: array
        Time, displacement, and velocity

    Examples:
    ----------
    >>> sdof_rk4(m=1, c=.1, k=1, x0=1, v0=0, n=8, dt=0.05)
    (array([ 0.  ,  0.05,  0.1 ,  0.15,  0.2 ,  0.25,  0.3 ,  0.35,  0.4 ]), array([[ 1.        ,  0.        ],
           [ 0.99875234, -0.04985443],
           [ 0.99502078, -0.0993359 ],
           [ 0.98882699, -0.14832292],
           [ 0.98019872, -0.19669582],
           [ 0.96916961, -0.24433704],
           [ 0.95577913, -0.29113145],
           [ 0.94007246, -0.33696662],
           [ 0.92210029, -0.38173305]]))
    """

    t = sp.linspace(0, n*dt, n+1)
    x = sp.zeros((n+1, 2))
    x[0, :] = x0, v0
    A = sp.array([[0, 1],
                  [-k/m, -c/m]])

    def f(x_): return A@x_

    for i in range(n):
        k1 = dt*f(x[i])
        k2 = dt*f(x[i] + k1/2)
        k3 = dt*f(x[i] + k2/2)
        k4 = dt*f(x[i] + k3)
        x[i+1] = x[i] + (k1 + 2.0*(k2+k3) + k4)/6.0

    return t, x


def euler_beam_modes(n=10, bctype=2, beamparams=sp.array((7.31e10, 1 / 12 * 0.03 * .015 ** 3, 2747, .015 * 0.03, 0.4)),
                     npoints=2001):
    """
    %VTB6_3 Natural frequencies and mass normalized mode shape for an Euler-
    % Bernoulli beam with a chosen boundary condition.
    % [w,x,U]=VTB6_3(n,bctype,bmpar,npoints) will return the nth natural 
    % frequency (w) and mode shape (U) of an Euler-Bernoulli beam.
    % If n is a vector, return the coresponding mode shapes and natural
    % frequencies.
    % With no output arguments the modes are ploted.
    % If only one mode is requested, and there are no output arguments, the
    % mode shape is animated.
    % The boundary condition is defined as follows:
    %
    % bctype = 1 free-free
    % bctype = 2 clamped-free
    % bctype = 3 clamped-pinned
    % bctype = 4 clamped-sliding
    % bctype = 5 clamped-clamped
    % bctype = 6 pinned-pinned
    %
    % The beam parameters are input through the vector bmpar:
    % bmpar = [E I rho A L];
    % where the variable names are consistent with Section 6.5 of the 
    % text.
    %
    %% Example: 20 cm long aluminum beam with h=1.5 cm, b=3 cm
    %% Animate the 4th mode for free-free boundary conditions
    % E=7.31e10;
    % I=1/12*.03*.015^3;
    % rho=2747;
    % A=.015*.03;
    % L=0.2;
    % vtb6_3(4,1,[E I rho A L]);
    %

    % Copyright Joseph C. Slater, 2007
    % Engineering Vibration Toolbox
    """
    E = beamparams[0]
    I = beamparams[1]
    rho = beamparams[2]
    A = beamparams[3]
    L = beamparams[4]
    if isinstance(n, int):
        ln = n
        n = sp.arange(n) + 1
    else:
        ln = len(n)

    # len=[0:(1/(npoints-1)):1]';  %Normalized length of the beam
    len = sp.linspace(0, 1, npoints)
    x = len * L
    # Determine natural frequencies and mode shapes depending on the
    # boundary condition.
    # Mass simplification. The following was arange_(1,length_(n)).reshape(-1)
    mode_num_range = sp.arange(0, ln)
    Bnl = sp.empty(ln)
    w = sp.empty(ln)
    U = sp.empty([npoints, ln])

    if bctype == 1:
        desc = 'Free-Free '
        Bnllow = sp.array((0, 0, 4.73004074486, 7.8532046241,
                           10.995607838, 14.1371654913, 17.2787596574))
        for i in mode_num_range:
            if n[i] > 7:
                Bnl[i] = (2 * n[i] - 3) * sp.pi / 2
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            if n[i] == 1:
                w[i] = 0
                U[:, i] = 1 + len * 0
            elif n[i] == 2:
                w[i] = 0
                U[:, i] = len - 0.5
            else:
                sig = (sp.cosh(Bnl[i]) - sp.cos(Bnl[i])) / \
                      (sp.sinh(Bnl[i]) - sp.sin(Bnl[i]))
                w[i] = (Bnl[i] ** 2) * sp.sqrt(E * I / (rho * A * L ** 4))
                b = Bnl[i] * len
                U[:, i] = sp.cosh(b) + sp.cos(b) - sig * \
                                                   (sp.sinh(b) + sp.sin(b))
    elif bctype == 2:
        desc = 'Clamped-Free '
        Bnllow = sp.array((1.88, 4.69, 7.85, 10.99, 14.14))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (2 * n[i] - 1) * sp.pi / 2
            else:
                Bnl[i] = Bnllow[i]

        for i in mode_num_range:
            sig = (sp.sinh(Bnl[i]) - sp.sin(Bnl[i])) / \
                  (sp.cosh(Bnl[i]) - sp.cos(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * sp.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * len
            # plt.plot(x,(sp.cosh(b) - sp.cos(b) - sig * (sp.sinh(b) - sp.sin(b))))
            U[:, i] = sp.cosh(b) - sp.cos(b) - sig * (sp.sinh(b) - sp.sin(b))

    elif bctype == 3:
        desc = 'Clamped-Pinned '
        Bnllow = sp.array((3.93, 7.07, 10.21, 13.35, 16.49))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (4 * n[i] + 1) * sp.pi / 4
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            sig = (sp.cosh(Bnl[i]) - sp.cos(Bnl[i])) / \
                  (sp.sinh(Bnl[i]) - sp.sin(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * sp.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * len
            U[:, i] = sp.cosh(b) - sp.cos(b) - sig * (sp.sinh(b) - sp.sin(b))
    elif bctype == 4:
        desc = 'Clamped-Sliding '
        Bnllow = sp.array((2.37, 5.5, 8.64, 11.78, 14.92))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (4 * n[i] - 1) * sp.pi / 4
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            sig = (sp.sinh(Bnl[i]) + sp.sin(Bnl[i])) / \
                  (sp.cosh(Bnl[i]) - sp.cos(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * sp.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * len
            U[:, i] = sp.cosh(b) - sp.cos(b) - sig * (sp.sinh(b) - sp.sin(b))
    elif bctype == 5:
        desc = 'Clamped-Clamped '
        Bnllow = sp.array((4.73, 7.85, 11, 14.14, 17.28))
        for i in mode_num_range:
            if n[i] > 4:
                Bnl[i] = (2 * n[i] + 1) * sp.pi / 2
            else:
                Bnl[i] = Bnllow[i]
        for i in mode_num_range:
            sig = (sp.cosh(Bnl[i]) - sp.cos(Bnl[i])) / \
                  (sp.sinh(Bnl[i]) - sp.sin(Bnl[i]))
            w[i] = (Bnl[i] ** 2) * sp.sqrt(E * I / (rho * A * L ** 4))
            b = Bnl[i] * len
            U[:, i] = sp.cosh(b) - sp.cos(b) - sig * (sp.sinh(b) - sp.sin(b))
    elif bctype == 6:
        desc = 'Pinned-Pinned '
        for i in mode_num_range:
            Bnl[i] = n[i] * sp.pi
            w[i] = (Bnl[i] ** 2) * sp.sqrt(E * I / (rho * A * L ** 4))
            U[:, i] = sp.sin(Bnl[i] * len)

    # Mass Normalization of mode shapes
    for i in mode_num_range:
        U[:, i] = U[:, i] / sp.sqrt(sp.dot(U[:, i], U[:, i]) * rho * A * L)

    """
    ppause=0
    x=len * L
    if nargout == 0:
        if length_(n) != 1:
            for i in arange_(1,length_(n)).reshape(-1):
                plot_(x,U[:,i])
                axis_([0,L,min_(min_(U)),max_(max_(U))])
                figure_(gcf)
                title_([desc,char('  '),char('Mode '),int2str_(i),char('     Natural Frequency = '),num2str_(w[i]),char(' rad/s')])
                ylabel_(char('Modal Amplitude'))
                xlabel_(char('Length along bar - x'))
                grid_(char('on'))
                disp_(char('Press return to continue'))
                pause
        else:
            nsteps=50
            clf
            step=2 * pi / (nsteps)
            i=arange_(0,(2 * pi - step),step)
            hold_(char('off'))
            handle=uicontrol_(char('style'),char('pushbutton'),char('units'),char('normal'),char('backgroundcolor'),char('red'),char('position'),[0.94,0.94,0.05,0.05],char('String'),char('Stop'),char('callback'),char('global stopstop;stopstop=1;'))
            handle2=uicontrol_(char('style'),char('pushbutton'),char('units'),char('normal'),char('backgroundcolor'),char('yellow'),char('position'),[0.94,0.87,0.05,0.05],char('String'),char('Pause'),char('callback'),char('global ppause;ppause=1;'))
            handle3=uicontrol_(char('style'),char('pushbutton'),char('units'),char('normal'),char('backgroundcolor'),char('green'),char('position'),[0.94,0.8,0.05,0.05],char('String'),char('Resume'),char('callback'),char('global ppause;ppause=0;'))
            stopstop=0
            bb=0
            while stopstop == 0 and bb < 100:

                bb=bb + 1
                for ii in [i].reshape(-1):
                    while ppause == 1:

                        pause_(0.01)
                        if stopstop == 1:
                            delete_(handle)
                            delete_(handle2)
                            delete_(handle3)
                            return w,x,U

                    plot_(x,U[:,1] * sp.cos(ii))
                    axis_([0,L,- max_(abs_(U)),max_(abs_(U))])
                    grid_(char('on'))
                    figure_(gcf)
                    title_([desc,char('  '),char('Mode '),int2str_(n),char('     \\omega_n = '),num2str_(w[1]),char(' rad/s')])
                    ylabel_(char('Modal Amplitude'))
                    xlabel_(char('Length along bar - x'))
                    drawnow

            clear_(char('stopstop'))
            delete_(handle)
            delete_(handle2)
            delete_(handle3)
    """
    return w, x, U


def euler_beam_frf(xin=0.22, xout=0.22, fmin=0.0, fmax=1000.0, zeta=0.02,
                   beamparams=sp.array((7.31e10, 1 / 12 * 0.03 * .015 ** 3, 2747, .015 * 0.03, 0.4)), bctype=2):
    print(fmin)
    E = beamparams[0]
    I = beamparams[1]
    rho = beamparams[2]
    A = beamparams[3]
    L = beamparams[4]
    np = 2001
    i = 0
    w = sp.linspace(fmin, fmax, 2001) * 2 * sp.pi
    if min([xin, xout]) < 0 or max([xin, xout]) > L:
        disp_(char('One or both locations are not on the beam'))
        return
    wn = sp.array((0, 0))
    # The number 100 is arbitrarily large and unjustified.
    a = sp.empty([np, 100], dtype=complex)
    f = sp.empty(100)

    while wn[-1] < 1.3 * (fmax * 2 * sp.pi):
        i = i + 1
        # legtext[i + 1]=[char('Contribution of mode '),num2str_(i)]
        wn, xx, U = euler_beam_modes(i, bctype, beamparams, 5000)
        spl = UnivariateSpline(xx, U[:, i - 1])
        Uin = spl(xin)
        Uout = spl(xout)
        # Uin=spline_(xx,U,xin)
        # Uout=spline_(xx,U,xout)

        # print(wn[-1])
        # print(w)
        a[:, i - 1] = rho * A * Uin * Uout / \
                      (wn[-1] ** 2 - w ** 2 + 2 * zeta * wn[-1] * w * sp.sqrt(-1))
        # print(a[0:10,i])
        # plt.plot(sp.log10(sp.absolute(a[:,i])))
        # input("Press Enter to continue...")
        f[i] = wn[-1] / 2 / sp.pi
    a = a[:, 0:i]
    plt.subplot(211)
    plt.plot(w / 2 / sp.pi, 20 * sp.log10(sp.absolute(sp.sum(a, axis=1))), '-')
    plt.hold('on')
    plt.plot(w / 2 / sp.pi, 20 * sp.log10(sp.absolute(a)), '-')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FRF (dB)')
    axlim = plt.axis()

    plt.axis(
        axlim + sp.array([0, 0, -0.1 * (axlim[3] - axlim[2]), 0.1 * (axlim[3] - axlim[2])]))

    plt.subplot(212)
    plt.plot(w / 2 / sp.pi, sp.unwrap(sp.angle(sp.sum(a, axis=1))) /
             sp.pi * 180, '-')
    plt.hold('on')
    plt.plot(w / 2 / sp.pi, sp.unwrap(sp.angle(a)) / sp.pi * 180, '-')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    axlim = plt.axis()
    plt.axis(
        axlim + sp.array([0, 0, -0.1 * (axlim[3] - axlim[2]), 0.1 * (axlim[3] - axlim[2])]))

    fout = w / 2 / sp.pi
    H = a
    return fout, H


def ebf(xin, xout, fmin, fmax, zeta):
    _, _ = euler_beam_frf(xin, xout, fmin, fmax, zeta)
    return


def ebf1(xin, xout):
    _, _ = euler_beam_frf(xin, xout)
    return


# def
# euler_beam_frf(xin=0.22,xout=0.22,fmin=0.0,fmax=1000.0,beamparams=sp.array((7.31e10,
# 1/12*0.03*.015**3, 2747, .015*0.03, 0.4)),


def frfplot(f, H):
    plt.subplot(211)
    plt.plot(f, 20 * sp.log10(sp.absolute(sp.sum(H, axis=1))), '-')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('FRF (dB)')
    axlim = plt.axis()
    plt.axis(
        axlim + sp.array([0, 0, -0.1 * (axlim[3] - axlim[2]), 0.1 * (axlim[3] - axlim[2])]))

    plt.subplot(212)
    plt.plot(f, sp.unwrap(sp.angle(sp.sum(H, axis=1))) / sp.pi * 180, '-')
    plt.grid('on')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    axlim = plt.axis()
    plt.axis(
        axlim + sp.array([0, 0, -0.1 * (axlim[3] - axlim[2]), 0.1 * (axlim[3] - axlim[2])]))


if __name__ == "__main__":
    import doctest
    import vtoolbox as vtb

    doctest.testmod(optionflags=doctest.ELLIPSIS)
    # doctest.run_docstring_examples(frfest,globals(),optionflags=doctest.ELLIPSIS)
    # doctest.run_docstring_examples(asd,globals(),optionflags=doctest.ELLIPSIS)
    """ What this does. 
    python (name of this file)  -v
    will test all of the examples in the help.

    Leaving off -v will run the tests without any output. Success will return nothing.

    See the doctest section of the python manual.
    https://docs.python.org/3.5/library/doctest.html
    """
