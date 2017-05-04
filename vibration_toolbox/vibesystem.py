import numpy as np
import scipy.linalg as la
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

__all__ = ['VibeSystem']

plt.style.use('seaborn-white')

color_palette = ["#4C72B0", "#55A868", "#C44E52",
                 "#8172B2", "#CCB974", "#64B5CD"]

plt.style.use({
    'lines.linewidth': 2.5,
    'axes.grid': True,
    'axes.linewidth': 0.1,
    'grid.color': '.9',
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.framealpha': 0.2
    })

colors = color_palette + [(.1, .1, .1)]
for code, color in zip('bgrmyck', colors):
    rgb = mpl.colors.colorConverter.to_rgb(color)
    mpl.colors.colorConverter.colors[code] = rgb
    mpl.colors.colorConverter.cache[code] = rgb


class VibeSystem(object):
    r"""A multiple degrees of freedom system.

    This class will create a multiple degree of
    freedom system given M, C and K matrices.

    Parameters
    ----------
    M : array
        Mass matrix.
    C : array
        Damping matrix.
    K : array
        Stiffness matrix.
    name : str, optional
        Name of the system.

    Attributes
    ----------
    evalues : array
        System's eigenvalues.
    evectors : array
        System's eigenvectors.
    wn : array
        System's natural frequencies in Hz.
    wd : array
        System's damped natural frequencies in Hz.
    H : scipy.signal.lti
        Continuous-time linear time invariant system

    Examples
    --------
    For a system consisting of two masses connected
    to each other and both connected to a wall we have
    the following matrices:

    >>> m1, m2 = 1, 1
    >>> c1, c2, c3 = 1, 1, 1
    >>> k1, k2, k3 = 1e3, 1e3, 1e3

    >>> M = np.array([[m1, 0],
    ...               [0, m2]])
    >>> C = np.array([[c1+c2, -c2],
    ...               [-c2, c2+c3]])
    >>> K = np.array([[k1+k2, -k2],
    ...               [-k2, k2+k3]])
    >>> sys = VibeSystem(M, C, K)
    >>> sys.wn
    array([ 5.03,  8.72])
    >>> sys.wd
    array([ 5.03,  8.71])
    """
    def __init__(self, M, C, K, name=None):
        self._M = M
        self._C = C
        self._K = K
        self.name = name
        # Values for evalues and evectors will be calculated by self._calc_system
        self.evalues = None
        self.evectors = None
        self.wn = None
        self.wd = None
        self.H = None

        self.n = len(M)
        self._calc_system()

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, value):
        self._M = value
        # if the parameter is changed this will update the system
        self._calc_system()

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value
        # if the parameter is changed this will update the system
        self._calc_system()

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value
        # if the parameter is changed this will update the system
        self._calc_system()

    def __repr__(self):
        M = np.array_str(self.M)
        K = np.array_str(self.K)
        C = np.array_str(self.C)
        return ('Mass Matrix: \n'
                '{} \n\n'
                'Stiffness Matrix: \n'
                '{} \n\n'
                'Damping Matrix: \n'
                '{}'.format(M, K, C))

    def _calc_system(self):
        self.evalues, self.evectors = self._eigen()
        self.wn = (np.absolute(self.evalues)/(2*np.pi))[:self.n]
        self.wd = (np.imag(self.evalues)/(2*np.pi))[:self.n]
        self.H = self._H()

    def A(self):
        """State space matrix
        
        This method will return the state space matrix
        of the system.
        
        Returns
        ----------
        A : array
            System's state space matrix. 

        Examples
        --------
        >>> m1, m2 = 1, 1
        >>> c1, c2, c3 = 1, 1, 1
        >>> k1, k2, k3 = 1e3, 1e3, 1e3

        >>> M = np.array([[m1, 0],
        ...               [0, m2]])
        >>> C = np.array([[c1+c2, -c2],
        ...               [-c2, c2+c3]])
        >>> K = np.array([[k1+k2, -k2],
        ...               [-k2, k2+k3]])
        >>> sys = VibeSystem(M, C, K) # create the system    
        >>> sys.A()
        array([[    0.,     0.,     1.,     0.],
               [    0.,     0.,     0.,     1.],
               [-2000.,  1000.,    -2.,     1.],
               [ 1000., -2000.,     1.,    -2.]])
        """

        Z = np.zeros((self.n, self.n))
        I = np.eye(self.n)

        A = np.vstack(
            [np.hstack([Z, I]),
             np.hstack([la.solve(-self.M, self.K),
                        la.solve(-self.M, self.C)])])
        return A

    @staticmethod
    def _index(eigenvalues):
        r"""Function used to generate an index that will sort
        eigenvalues and eigenvectors based on the imaginary (wd)
        part of the eigenvalues. Positive eigenvalues will be
        positioned at the first half of the array.
        """
        # avoid float point errors when sorting
        evals_truncated = np.around(eigenvalues, decimals=10)
        a = np.imag(evals_truncated)  # First column
        b = np.absolute(evals_truncated)  # Second column
        ind = np.lexsort((b, a))  # Sort by imag, then by absolute
        # Positive eigenvalues first
        positive = [i for i in ind[len(a) // 2:]]
        negative = [i for i in ind[:len(a) // 2]]

        idx = np.array([positive, negative]).flatten()

        return idx

    def _eigen(self, sorted_=True):
        r"""This method will return the eigenvalues and eigenvectors of
        the state space matrix A, sorted by the index method which
        considers the imaginary part (wd) of the eigenvalues for sorting.
        To avoid sorting use sorted_=False
        """
        evalues, evectors = la.eig(self.A())
        if sorted_ is False:
            return evalues, evectors

        idx = self._index(evalues)

        return evalues[idx], evectors[:, idx]

    def _H(self):
        r"""Continuous-time linear time invariant system.

        This method is used to create a Continuous-time linear
        time invariant system for the mdof system.
        From this system we can obtain poles, impulse response,
        generate a bode, etc.
        """
        Z = np.zeros((self.n, self.n))
        I = np.eye(self.n)

        # x' = Ax + Bu
        B2 = I
        A = self.A()
        B = np.vstack([Z,
                       la.solve(self.M, B2)])

        # y = Cx + Du
        # Observation matrices
        Cd = I
        Cv = Z
        Ca = Z

        C = np.hstack((Cd - Ca @ la.solve(self.M, self.K),
                       Cv - Ca @ la.solve(self.M, self.C)))
        D = Ca @ la.solve(self.M, B2)

        sys = signal.lti(A, B, C, D)

        return sys

    def time_response(self, F, t, ic=None):
        r"""Time response for a mdof system.

        This method returns the time response for a mdof system
        given a force, time and initial conditions.

        Parameters
        ----------
        F : array
            Force array (needs to have the same length as time array).
        t : array
            Time array.
        ic : array, optional
            The initial conditions on the state vector (zero by default).

        Returns
        ----------
        t : array
            Time values for the output.
        yout : array
            System response.
        xout : array
            Time evolution of the state vector.


        Examples
        --------
        >>> m1, m2 = 1, 1
        >>> c1, c2, c3 = 1, 1, 1
        >>> k1, k2, k3 = 1e3, 1e3, 1e3

        >>> M = np.array([[m1, 0],
        ...               [0, m2]])
        >>> C = np.array([[c1+c2, -c2],
        ...               [-c2, c2+c3]])
        >>> K = np.array([[k1+k2, -k2],
        ...               [-k2, k2+k3]])
        >>> sys = VibeSystem(M, C, K) # create the system
        >>> t = np.linspace(0, 25, 1000) # time array
        >>> F2 = np.zeros((len(t), 2))
        >>> F2[:, 1] = 1000*np.sin(40*t) # force applied on m2
        >>> t, yout, xout = sys.time_response(F2, t)
        >>> # response on m1
        >>> yout[:5, 0]
        array([ 0.  ,  0.  ,  0.07,  0.32,  0.61])
        >>> # response on m2 
        >>> yout[:5, 1]
        array([ 0.  ,  0.08,  0.46,  0.79,  0.48])
        """
        if ic is not None:
            return signal.lsim(self.H, F, t, ic)
        else:
            return signal.lsim(self.H, F, t)

    def freq_response(self, omega=None, modes=None):
        r"""Frequency response for a mdof system.

        This method returns the frequency response for a mdof system
        given a range of frequencies and the modes that will be used.

        Parameters
        ----------
        omega : array, optional
            Array with the desired range of frequencies (the default
             is 0 to 1.5 x highest damped natural frequency.
        modes : list, optional
            Modes that will be used to calculate the frequency response
            (all modes will be used if a list is not given).

        Returns
        ----------
        omega : array
            Array with the frequencies
        magdb : array
            Magnitude (dB) of the frequency response for each pair input/output.
            The order of the array is: [output, input, magnitude]
        phase : array
            Phase of the frequency response for each pair input/output.
            The order of the array is: [output, input, phase]

        Examples
        --------
        >>> m1, m2 = 1, 1
        >>> c1, c2, c3 = 1, 1, 1
        >>> k1, k2, k3 = 1e3, 1e3, 1e3

        >>> M = np.array([[m1, 0],
        ...               [0, m2]])
        >>> C = np.array([[c1+c2, -c2],
        ...               [-c2, c2+c3]])
        >>> K = np.array([[k1+k2, -k2],
        ...               [-k2, k2+k3]])
        >>> sys = VibeSystem(M, C, K) # create the system
        >>> omega, magdb, phase = sys.freq_response()
        >>> # magnitude for output on 0 and input on 1.
        >>> print(np.array_str(magdb[0, 1, :4], precision=2)) 
        [-69.54 -69.54 -69.54 -69.54]
        >>> # phase for output on 1 and input on 1.
        >>> print(np.array_str(phase[1, 1, :4], precision=5, suppress_small=True)) 
        [...0.      -0.00471 -0.00942 -0.01413] 
        """
        rows = self.H.inputs  # inputs (mag and phase)
        cols = self.H.inputs  # outputs

        B = self.H.B
        C = self.H.C
        D = self.H.D

        evals = self.evalues
        psi = self.evectors
        psi_inv = la.inv(psi)  # TODO change to get psi_inv from la.eig

        # if omega is not given, define a range
        if omega is None:
            omega = np.linspace(0, max(evals.imag) * 1.5, 1000)

        # if modes are selected:
            if modes is not None:
                n = self.n  # n dof -> number of modes
                m = len(modes)  # -> number of desired modes
                # idx to get each evalue/evector and its conjugate
                idx = np.zeros((2 * m), int)
                idx[0:m] = modes  # modes
                idx[m:] = range(2 * n)[-m:]  # conjugates (see how evalues are ordered)

                evals_m = evals[np.ix_(idx)]
                psi_m = psi[np.ix_(range(2 * n), idx)]
                psi_inv_m = psi_inv[np.ix_(idx, range(2 * n))]

                magdb_m = np.empty((cols, rows, len(omega)))
                phase_m = np.empty((cols, rows, len(omega)))

                for wi, w in enumerate(omega):
                    diag = np.diag([1 / (1j * w - lam) for lam in evals_m])
                    H = C @ psi_m @ diag @ psi_inv_m @ B + D

                    magh = 20.0 * np.log10(abs(H))
                    angh = np.rad2deg((np.angle(H)))

                    magdb_m[:, :, wi] = magh
                    phase_m[:, :, wi] = angh

                return omega, magdb_m, phase_m

        magdb = np.empty((cols, rows, len(omega)))
        phase = np.empty((cols, rows, len(omega)))

        for wi, w in enumerate(omega):
            diag = np.diag([1 / (1j * w - lam) for lam in evals])
            H = C @ psi @ diag @ psi_inv @ B + D

            magh = 20.0 * np.log10(abs(H))
            angh = np.rad2deg((np.angle(H)))

            magdb[:, :, wi] = magh
            phase[:, :, wi] = angh

        return omega, magdb, phase

    def plot_freq_response(self, out, inp, ax0=None, ax1=None):
        """Plot frequency response.
        
        This method plots the frequency response given
        an output and an input.

        Parameters
        ----------
        out : int
            Output.
        input : int
            Input.
        
        ax0 : matplotlib.axes, optional
            Matplotlib axes where the amplitude will be plotted.
            If None creates a new.
        ax1 : matplotlib.axes, optional
            Matplotlib axes where the phase will be plotted.
            If None creates a new.

        Returns
        -------
        ax0 : matplotlib.axes
            Matplotlib axes with amplitude plot.
        ax1 : matplotlib.axes
            Matplotlib axes with phase plot.
            
        Examples
        --------
        >>> m1, m2 = 1, 1
        >>> c1, c2, c3 = 1, 1, 1
        >>> k1, k2, k3 = 1e3, 1e3, 1e3

        >>> M = np.array([[m1, 0],
        ...               [0, m2]])
        >>> C = np.array([[c1+c2, -c2],
        ...               [-c2, c2+c3]])
        >>> K = np.array([[k1+k2, -k2],
        ...               [-k2, k2+k3]])
        >>> sys = VibeSystem(M, C, K) # create the system
        >>> # plot frequency response for input and output at m1
        >>> sys.plot_freq_response(0, 0)
        (<matplotlib.axes._...
        """
        if ax0 is None or ax1 is None:
            fig, ax = plt.subplots(2)
            if ax0 is not None:
                _, ax1 = ax
            if ax1 is not None:
                ax0, _ = ax
            else:
                ax0, ax1 = ax

        omega, magdb, phase = self.freq_response()

        ax0.plot(omega, magdb[out, inp, :])
        ax1.plot(omega, phase[out, inp, :])
        for ax in [ax0, ax1]:
            ax.set_xlim(0, max(omega))
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(prune='lower'))
            ax.yaxis.set_major_locator(
                mpl.ticker.MaxNLocator(prune='upper'))

        ax0.text(.9, .9, 'Output %s' % out,
                 horizontalalignment='center',
                 transform=ax0.transAxes)
        ax0.text(.9, .7, 'Input %s' % inp,
                 horizontalalignment='center',
                 transform=ax0.transAxes)

        ax0.set_ylabel('Magnitude $(dB)$')
        ax1.set_ylabel('Phase')
        ax1.set_xlabel('Frequency (rad/s)')

        return ax0, ax1

    def plot_freq_response_grid(self, outs, inps, ax=None):
        """Plot frequency response.
        
        This method plots the frequency response given
        an output and an input.

        Parameters
        ----------
        outs : list
            List with the desired outputs.
        inps : list
            List with the desired outputs.        
        
        ax : array with matplotlib.axes, optional
            Matplotlib axes array created with plt.subplots.
            It needs to have a shape of (2*inputs, outputs).

        Returns
        -------
        ax : array with matplotlib.axes, optional
            Matplotlib axes array created with plt.subplots.
           
        Examples
        --------
        >>> m1, m2 = 1, 1
        >>> c1, c2, c3 = 1, 1, 1
        >>> k1, k2, k3 = 1e3, 1e3, 1e3

        >>> M = np.array([[m1, 0],
        ...               [0, m2]])
        >>> C = np.array([[c1+c2, -c2],
        ...               [-c2, c2+c3]])
        >>> K = np.array([[k1+k2, -k2],
        ...               [-k2, k2+k3]])
        >>> sys = VibeSystem(M, C, K) # create the system
        >>> # plot frequency response for inputs at [0, 1]
        >>> # and outputs at [0, 1] 
        >>> sys.plot_freq_response_grid(outs=[0, 1], inps=[0, 1])
        array([[<matplotlib.axes._...
        """
        if ax is None:
            fig, ax = plt.subplots(len(inps) * 2, len(outs),
                                   sharex=True,
                                   figsize=(4*len(outs), 3*len(inps)))
            fig.subplots_adjust(hspace=0.001, wspace=0.25)

        if len(outs) > 1:
            for i, out in enumerate(outs):
                for j, inp in enumerate(inps):
                    self.plot_freq_response(out, inp,
                                            ax[2*i, j],
                                            ax[2*i + 1, j])
        else:
            for i, inp in enumerate(inps):
                self.plot_freq_response(outs[0], inp,
                                        ax[2*i],
                                        ax[2*i + 1])

        return ax
