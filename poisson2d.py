import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """
    


    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xij, self.yij ...
        self.N = N
        self.h = self.L / N
        x = np.linspace(0, self.L, N+1)
        y = np.linspace(0, self.L, N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')

    def D2(self):
        """Return second order differentiation matrix"""
        N = self.N
        h = self.h
        main_diag = -2.0 *np.ones(N+1)
        off_diag = 1.0 * np.ones(N)
        D2 = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N+1, N+1)) / h**2
        return D2

    def laplace(self):
        """Return vectorized Laplace operator"""
        D2 = self.D2()
        I = sparse.identity(self.N+1)
        L = sparse.kron(D2, I) + sparse.kron(I, D2)
        return L


    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        N = self.N
        boundary = []
        for i in range(N+1):
            for j in range(N+1):
                if i == 0 or i == N or j == 0 or j == N:
                    idx = i * (N+1) + j 
                    boundary.append(idx)
        return np.array(boundary)


    def assemble(self):
        """Return assembled matrix A and right hand side vector b"""
        # return A, b
        N = self.N
        L = self.laplace()
        xij = self.xij
        yij = self.yij
        f = self.f

        F = np.zeros((N+1, N+1))
        for i in range(N+1):
            for j in range(N+1):
                F[i, j] = float(f.subs({x: xij[i, j], y: yij[i,j]}))

        b = F.flatten()
        A = L.tolil()

        boundary = self.get_boundary_indices()
        for idx in boundary: 
            i, j = divmod(idx, N+1)
            A[idx, :] = 0
            A[idx, idx] = 1
            b[idx] = float(self.ue.sub({x: xij[i, j], y: yij[i, j]}))

        return A.tocsr(), b
    

    def l2_error(self, u):
        """Return l2-error norm"""
        exact = np.array([[float(self.ue.subs({x: self.xij[i, j], y: self.yij[i, j]}))
                           for j in range(self.N+1)] for i in range(self.N+1)])
        error = u - exact
        return np.sqrt(np.sum(error**2) * self.h**2)

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

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
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        from scipy.interpolate import RectBivariateSpline
        interpolator = RectBivariateSpline(
            np.linspace(0, self.L, self.N+1),
            np.linspace(0, self.L, self.N+1),
            self.U)
        return float(interpolator(x, y))

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3
