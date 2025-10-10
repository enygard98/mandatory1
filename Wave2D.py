import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N+1, N+1), format='lil')
        D[0, :] = 0
        D[-1, :] = 0
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.mx, self.my = mx, my
        self.create_mesh(N)
        
        xij, yij = self.xij, self.yij
        w = self.c * np.pi * np.sqrt(mx**2 + my**2)
        Unm1 = np.sin(mx * np.pi * xij) * np.sin(my * np.pi * yij)  # t=0
        dt = self.dt
        Un = np.sin(mx * np.pi * xij) * np.sin(my * np.pi * yij) * np.cos(w * (-dt))  # t=-dt
        
        for arr in [Unm1, Un]:
            arr[0, :] = 0
            arr[-1, :] = 0
            arr[:, 0] = 0
            arr[:, -1] = 0
        return Unm1, Un

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * (1.0 / self.N) / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        w = self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)
        xij, yij = self.xij, self.yij
        u_exact = np.sin(self.mx * np.pi * xij) * np.sin(self.my * np.pi * yij) * np.cos(w * t0)
        error = u - u_exact
       
        interior = error[1:-1, 1:-1]
        h = 1.0 / self.N
        l2 = np.sqrt(np.sum(interior**2) * h**2)
        return l2

    def apply_bcs(self, U):
        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 0
        U[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        # Setup mesh and initial condition
        self.create_mesh(N)
        xij, yij = self.xij, self.yij
        dx = 1.0 / N
        D = self.D2(N) / dx**2
        dt = self.dt

        Unm1, Un = self.initialize(N, mx, my)

        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()

        Unp1 = np.zeros_like(Un)
        for n in range(2, Nt+1):
            Unp1[:] = 2 * Un - Unm1 + (c * dt)**2 * (D @ Un + Un @ D.T)
            self.apply_bcs(Unp1)
            # Swap solutions for next step
            Unm1, Un = Un, Unp1
            if store_data == 1:
                plotdata[n] = Un.copy()
        if store_data == 1:
            return xij, yij, plotdata
        else:
            # Return l2-errors for each step, like your original API
            errors = [self.l2_error(arr, k*dt) for k, arr in plotdata.items()]
            return dx, np.array(errors)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        raise NotImplementedError

    def ue(self, mx, my):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print("Convergence rates:", r)
    print("Errors:", E)
    print("Mesh sizes:", h)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    raise NotImplementedError

    solN = Wave2D_Neumann()
    hN, errorsN = solN(N, Nt, cfl = cfl, mx = mx, my = my, store_data = -1)
    assert np.all(errorsN < 1e-12), f"Neumann errors not small: {errorsN}"
