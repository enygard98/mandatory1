import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')

    def D2(self, N):
        """Return second order differentiation matrix"""
        h = 1.0 / N
        main_diag = -2.0 * np.ones(N + 1)
        off_diag = np.ones(N)
        D2 = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        D2[0, :] = 0.0
        D2[-1, :] = 0.0
        D2[0, 0] = 1.0
        D2[-1, -1] = 1.0
        return D2 / h**2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.pi * np.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        if hasattr(self, 'neumann') and self.neumann:
            return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
        else:
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
        self.create_mesh(N)
        if hasattr(self, 'neumann') and self.neumann:
            ue = lambda x, y, t: np.cos(mx*np.pi*x)*np.cos(my*np.pi*y)*np.cos(self.w*t)
        else:
            ue = lambda x, y, t: np.sin(mx*np.pi*x)*np.sin(my*np.pi*y)*np.cos(self.w*t)
        u0 = ue(self.xij, self.yij, 0)
        u1 = ue(self.xij, self.yij, -self.dt)
        return u0, u1

    @property
    def dt(self):
        """Return the time step"""
        h = 1.0 / self.N
        return self.cfl * h / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        if hasattr(self, 'neumann') and self.neumann:
            ue = lambda x, y: np.cos(self.mx*np.pi*x)*np.cos(self.my*np.pi*y)*np.cos(self.w*t0)
        else:
            ue = lambda x, y: np.sin(self.mx*np.pi*x)*np.sin(self.my*np.pi*y)*np.cos(self.w*t0)
        u_exact = ue(self.xij, self.yij)
        return np.sqrt(np.sum((u - u_exact)**2) / u.size)

    def apply_bcs(self, u):
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        return u

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

        h = 1.0 / N
        dt = self.dt

        u0, u1 = self.initialize(N, mx, my)
        u_nm1 = u1.copy()
        u_n = u0.copy()

        D2 = self.D2(N)
        # Kronecker sum for Laplacian
        I = np.eye(N+1)
        L = np.kron(D2, I) + np.kron(I, D2)

        u_list = []
        err_list = []
        for n in range(Nt):
            U = u_n.flatten()
            U_nm1 = u_nm1.flatten()
            # Leapfrog scheme
            U_np1 = 2*U - U_nm1 + (dt**2)*(self.c**2)*(L @ U)
            u_np1 = U_np1.reshape((N+1, N+1))
            u_np1 = self.apply_bcs(u_np1)
            if store_data > 0 and n % store_data == 0:
                u_list.append(u_np1.copy())
            if store_data == -1:
                err = self.l2_error(u_np1, (n+1)*dt)
                err_list.append(err)
            u_nm1 = u_n
            u_n = u_np1

        if store_data > 0:
            return {k: v for k, v in enumerate(u_list)}
        else:
            return h, err_list

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
    neumann = True

    def D2(self, N):
        h = 1.0 / N
        main_diag = -2.0 * np.ones(N + 1)
        off_diag = np.ones(N)
        D2 = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        # Neumann BC: set first and last rows for zero derivative (forward/backward difference)
        D2[0, 0] = -2.0
        D2[0, 1] = 2.0
        D2[0, 2:] = 0.0

        D2[-1, -1] = -2.0
        D2[-1, -2] = 2.0
        D2[-1, :-2] = 0.0

        return D2 / h**2


    def apply_bcs(self, u):
        # Neumann BC: zero-gradient at boundaries
        u[0, :] = u[2, :]
        u[-1, :] = u[-3, :]
        u[:, 0] = u[:, 2]
        
        return u

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    print("Dirichlet convergence rates:", r)
    print("Dirichlet errors:", E)
    print("Dirichlet mesh sizes:", h)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    print("Neumann convergence rates:", r)
    print("Neumann errors:", E)
    print("Neumann mesh sizes:", h)
    assert abs(r[-1]-2) < 1.5

def test_exact_wave2d():
    sol = Wave2D()
    N = 32
    Nt = 2
    h, err = sol(N, Nt, cfl=0.1, mx=3, my=3, store_data=-1)
    assert err[0] < 1e-5


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed!")
