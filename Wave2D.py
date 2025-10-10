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
        self.N = N
        self.h = 1.0 / N
        x = np.linspace(0, 1, N+1)
        y = np.linspace(0, 1, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')

    def D2(self, N):
        """Return second order differentiation matrix
        not needed for the finite difference implementation"""
        h = 1.0 / N
        main_diag = -2.0 * np.ones(N-1)
        off_diag = 1.0 * np.ones(N-2)
        D2 = (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)) / h**2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)
   

    def ue(self, mx, my):
        """Return the exact standing wave"""
        def sol(xv, yv, tv):
            return np.sin(mx*np.pi*xv) * np.sin(my*np.pi*yv) * np.cos(self.w * tv)
        return sol

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.create_mesh(N)
        self.mx = mx
        self.my = my
        u_exact = self.ue(mx, my)
        U = u_exact(self.xij, self.yij, 0)
        U_prev = u_exact(self.xij, self.yij, -self.dt)
        U[0, :] = 0; U[-1, :] =0; U[:, 0] = 0; U[:, -1] = 0
        U_prev[0, :] = 0; U_prev[-1, :] = 0; U_prev[:, 0] = 0; U_prev[:, -1] = 0
        return U, U_prev


    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.h / self.c 

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_exact = self.ue(self.mx, self.my)(self.xij, self.yij, t0)
        error = u - u_exact
        interior = error[1: -1, 1: -1]
        l2 = np.sqrt(np.sum(interior**2) * self.h**2)
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
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        self.create_mesh(N)
        h = self.h
        dt = self.dt
        
        U, U_prev = self.initialize(N, mx, my)
        data = {} if store_data > 0 else []

        for k in range(Nt):
            laplacian = (U[2:, 1:-1] + U[:-2, 1:-1] + U[1:-1, 2:] + U[1:-1, :-2] - 4 * U[1:-1, 1:-1]) / h**2
            U_np1 = np.zeros_like(U)
            U_np1[1:-1, 1:-1] = (2 * U[1:-1, 1:-1] - U_prev[1:-1, 1:-1] + (dt**2) * self.c**2 * laplacian)  
            self.apply_bcs(U_np1)
            
            if store_data > 0 and k % store_data == 0:
                data[k] = U_np1.copy()
            elif store_data == -1:
                data.append(self.l2_error(U_np1, (k+1)*dt))             
            U_prev, U = U, U_np1
        
        if store_data > 0:
            return data
        else:
            return h, np.array(data)

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
        h = 1.0 / N
        D2 = np.zeros((N+1, N+1))

        for i in range(1, N): 
            D2[i, i-1] = 1.0
            D2[i, i] = -2.0
            D2[i, i+1] = 1.0

        D2[0, 0] = -2.0
        D2[0, 1] = 2.0
        D2[N, N] = -2.0
        D2[N, N-1] = 2.0
        D2 = D2 / h**2
        return D2

    def ue(self, mx, my):
        
        def sol(xv, yv, tv):
            return np.cos(mx * np.pi * xv) * np.cos(my * np.pi * yv) * np.cos(self.w * tv)
        return sol

    def apply_bcs(self, U):
    
        U[0, :] = U[1, :]      
        U[-1, :] = U[-2, :]     
        U[:, 0] = U[:, 1]      
        U[:, -1] = U[:, -2]  

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    N = 32
    Nt = 10
    cfl = 0.1
    mx, my = 2, 3
    h, errors = sol(N, Nt, cfl = cfl, mx = mx, my = my, store_data = -1)

    assert np.all(errors < 1e-12), f"Dirichlet errors not small: {errors}"

    solN = Wave2D_Neumann()
    hN, errorsN = solN(N, Nt, cfl = cfl, mx = mx, my = my, store_data = -1)
    assert np.all(errorsN < 1e-12), f"Neumann errors not small: {errorsN}"
