import numpy as np
import polars as pl
from icecream import ic
import tensorflow as tf
from scipy.interpolate import interp1d
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar


class GaussianIntegral:
    """
    A utility class to compute gaussian integrals of functions of the form
    x -> f(mu + sigma * x) for a given function f. The integrals are precomputed
    on a grid and then interpolated using scipy's RegularGridInterpolator. For
    each function f, the computation is done only once and then cached. Besides
    the function f, the class also stores the integral of f' and f^2. The class
    is callable and returns the interpolated value of the integral for a given
    mu and sigma.

    Parameters
    ----------
    mu : tuple
        The range of mu values to consider
    sigma : tuple
        The range of sigma values to consider
    n_points : int
        The number of points to use for the grid
    """

    def __init__(self, sigma=(1e-8, 100), n_points=10_000):
        self.cache = {}
        self.sigma = sigma
        self.n_points = n_points

    def __call__(self, f, sigma=1, function="f"):
        """
        Compute the integral of f(sigma * x) for a given mu and sigma. The
        function f is assumed to be vectorized, i.e. it should accept an array
        of x values and return an array of f values. The integral is computed
        using scipy's quad function. The result is cached and interpolated using
        scipy's RegularGridInterpolator.

        Parameters
        ----------
        f : callable
            The function to integrate. Has to be a vectorized function with a name!
        mu : iterable or number (if iterable, should have the same length as sigma)
            The mean of the gaussian
        sigma : iterable or number (if iterable, should have the same length as mu)
            The standard deviation of the gaussian
        function : str
            The function to return. Can be 'f', 'fp' or 'f2'

        Returns
        -------
        float or array
            The integral of f(mu + sigma * x) for the given mu and sigma
        """
        if function not in ["f", "fp", "f2"]:
            raise ValueError("Invalid function")
        if f.__name__ not in self.cache:
            ic("Computing integral for", f.__name__)
            sigmas = np.linspace(*self.sigma, self.n_points)

            def gaussint(fun):
                g = (lambda x: x.numpy()) if isinstance(fun(0), tf.Tensor) else lambda x: x

                return quad_vec(
                    lambda x: g(fun(x)) * np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi),
                    -np.inf,
                    np.inf,
                )[0]

            self.cache[f.__name__] = {
                "f": interp1d(sigmas, gaussint(lambda x: f(x * sigmas))),
                "fp": interp1d(sigmas, gaussint(lambda x: x * f(x * sigmas) / sigmas)),
                "f2": interp1d(sigmas, gaussint(lambda x: f(x * sigmas) ** 2)),
            }
            ic("Done computing integral for", f.__name__)

        return self.cache[f.__name__][function](sigma)


gaussian_integral = GaussianIntegral()


class SCov:
    """
    Deterministic equivalents to sample covariance matrices with population covariance matrix Omega

    Parameters
    ----------
    Omega : np.ndarray, shape=(p,p)
        The population covariance matrix
    eps : float
        The precision of the iterative algorithm to compute m(l,n)

    Attributes
    ----------
    OmegaEig : np.ndarray, shape=(p,)
        The eigenvalues of the population covariance matrix
    U : np.ndarray, shape=(p,p)
        The eigenvectors of the population covariance matrix

    Methods
    ----------
    m(l,n=n,m0=1) : float
        The deterministic equivalent to the averaged trace of the Gram resolvent $\check{G}$
        with regularization parameter $l$ and $n$ samples
    M(l,n=n) : np.ndarray, shape=(p,p)
        The deterministic equivalent to the resolvent $G$ with regularization parameter $l$ and $n$ samples
    """

    def __init__(self, Omega, eps=1e-8):
        assert Omega.shape[0] == Omega.shape[1] and len(Omega.shape) == 2
        self.Omega = Omega
        self.OmegaEig, self.U = np.linalg.eigh(Omega)
        self.p = Omega.shape[0]
        self.eps = eps
        self.m_cache = {}

    def _m(self, lamb, n=None, history=False, m0=1):
        n = self.p if n is None else n

        def iter_m(m):
            return 1 / (lamb + np.sum(self.OmegaEig / (n * m * self.OmegaEig + 1)))

        m1 = iter_m(m0)
        hist = [abs(m0 - m1)]
        while abs(m0 - m1) > self.eps:
            m0, m1 = m1, iter_m(m1)
            if history:
                hist.append(abs(m0 - m1))
        return [m1, hist] if history else m1

    def m(self, lamb, n=None, m0=None, verbose=False):
        """
        m(l,n=n) is the solution to the equation m = 1/(l+ <Omega@inv(1 + n/p*Omega*l*m)>)

        Parameters
        ----------
        l : float
            The regularization parameter
        n : int
            The number of samples
        history : bool
            Whether to return the history of the iterative algorithm
        m0 : float
            The initial guess for the iterative algorithm

        Returns
        -------
        float or list
            The solution to the equation m = 1/(l+ <Omega@inv(1 + n/p*Omega*l*m)>) or [m, history] if history=True
        """
        n = self.p if n is None else n
        if (lamb, n) in self.m_cache:
            if verbose:
                ic("Using cached value")
            return self.m_cache[(lamb, n)]
        else:
            if self.m_cache and not m0:
                closest_ln = min(
                    self.m_cache.keys(),
                    key=lambda x: (abs(x[0] - lamb) / lamb) ** 2 + (abs(x[1] - n) / n) ** 2,
                )
                m0 = self.m_cache[closest_ln]
                if verbose:
                    ic("Starting from cache", m0, closest_ln)
            elif not m0:
                if verbose:
                    ic("Starting from m0 = 1")
                m0 = 1
            (m, hist) = self._m(lamb, n=n, m0=m0, history=True)
            if verbose:
                ic(m0, m, len(hist))
            self.m_cache[(lamb, n)] = m
            return m

    def M(self, lamb, n=None, eigs=False):
        """
        M(l,n=n) is the deterministic equivalent to the resolvent $G$ with parameter $l$ and $n$ samples

        Parameters
        ----------
        l : float
            The regularization parameter
        n : int
            The number of samples
        eigs : bool
            Whether to return the eigenvalues of the resolvent or the resolvent itself

        Returns
        -------
        np.ndarray, shape=(p,p) or np.ndarray, shape=(p,)
            The deterministic equivalent to the resolvent $G$ with parameter $l$ and $n$ samples or its eigenvalues
        """
        n = self.p if n is None else n
        if eigs:
            return 1 / (n * lamb * self.m(lamb, n=n) * self.OmegaEig + lamb)
        else:
            return self.U @ np.diag(self.M(lamb, n=n, eigs=True)) @ self.U.T


class FeatureRidgeRegression:
    """
    A class to compute the generalization error of ridge regression with a given population covariance matrix Omega, label variance sigma**2 and label-feature covariance psi. If data is provided, the class can also compute the ridge regression vector. If test_data is provided, the class can also compute the empirical generalization error of the learned ridge regression vector.

    Parameters
    ----------
    Omega : np.ndarray, shape=(p,p)
        The population covariance matrix
    sigma : float
        The label variance
    psi : np.ndarray, shape=(p,)
        The label-feature covariance
    data : tuple
        A tuple (Phi, y) where Phi is the feature matrix of shape (p,n) and y is the target vector of shape (n,)
    test_data : tuple
        A tuple (Phi_test, y_test) where Phi_test is the test feature matrix of shape (p,n_test) and y_test is the test target vector of shape (n_test,)

    Methods
    ----------
    learningCurveEmp(l, n=None, repeats=1) : pl.DataFrame
        Compute the empirical generalization error of the learned ridge regression vector for different numbers of samples and regularization parameters. Requires data and test_data to be provided.
    genErrRMT(l, n=None) : float
        Compute the generalization error of ridge regression with regularization parameter l and n samples. Requires the population covariance matrix, label variance and label-feature covariance to be provided.
    learningCurveRMT(l, n=None) : pl.DataFrame
        Compute the generalization error of ridge regression with regularization parameter l and n samples for different numbers of samples and regularization parameters. Requires the population covariance matrix, label variance and label-feature covariance to be provided.
    """

    def __init__(self, Omega=None, sigma=None, psi=None, data=None, test_data=None, **kwargs):
        self.Omega, self.psi, self.sigma = Omega, psi, sigma
        assert self.Omega is not None and self.psi is not None and self.sigma is not None
        assert self.Omega.shape[0] == self.Omega.shape[1] == len(self.psi)
        if data is not None:
            self.Phi = data[0]
            self.y = data[1]
        if test_data is not None:
            self.Phi_test = test_data[0]
            self.y_test = test_data[1]
        self.p = self.Omega.shape[0]
        self.sc = SCov(self.Omega, **kwargs)
        self.UTpsi = self.sc.U.T @ self.psi

    @classmethod
    def empirical(cls, model, train_data=None, test_data=None, emp_avg_data=None, center=True):
        fmodel = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        features = {
            name: fmodel(data[0]).numpy()
            for (name, data) in [
                ("train", train_data),
                ("test", test_data),
                ("emp_avg", emp_avg_data),
            ]
        }
        y = {
            name: data[1]
            for (name, data) in [
                ("train", train_data),
                ("test", test_data),
                ("emp_avg", emp_avg_data),
            ]
        }
        if center:
            for name in ["train", "test", "emp_avg"]:
                features[name] -= features["emp_avg"].mean(axis=0, keepdims=True)
        Omega = features["emp_avg"].T @ features["emp_avg"] / features["emp_avg"].shape[0]
        psi = (features["emp_avg"] * y["emp_avg"][:, np.newaxis]).mean(axis=0)
        sigma = np.mean(y["emp_avg"] ** 2) ** 0.5
        return cls(
            Omega=Omega,
            psi=psi,
            sigma=sigma,
            data=(features["train"].T, y["train"]),
            test_data=(features["test"].T, y["test"]),
        )

    @classmethod
    def linearized(cls, model, train_data=None, test_data=None, emp_avg_data=None, avg=True):
        fmodel = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        features = {name: fmodel(data[0]).numpy() for (name, data) in [("train", train_data), ("test", test_data)]}
        sigma = np.mean(emp_avg_data[1] ** 2) ** 0.5
        x = emp_avg_data[0].reshape(emp_avg_data[0].shape[0], -1)
        Omega0 = x.T @ x / x.shape[0]
        psi = np.mean(x * emp_avg_data[1][:, np.newaxis], axis=0)

        for i in range(2, len(fmodel.layers)):
            try:
                f = fmodel.layers[i].activation
                W = fmodel.layers[i].weights[0].numpy()
            except AttributeError:
                raise ValueError("Unsupported Architecture")
            Omega1 = W.T @ Omega0 @ W
            if avg:
                std = np.mean(np.diag(Omega1)) ** 0.5
                fp = gaussian_integral(f, sigma=std, function="fp")
                f2 = gaussian_integral(f, sigma=std, function="f2")
                Omega0 = Omega1 * fp**2 + np.identity(W.shape[1]) * (f2 - fp**2 * std**2)
                psi = fp * W.T @ psi
            else:
                std = np.diag(Omega1) ** 0.5
                fp = gaussian_integral(f, sigma=std, function="fp")
                f2 = gaussian_integral(f, sigma=std, function="f2")
                Omega0 = np.diag(fp) @ (Omega1 - np.diag(np.diag(Omega1))) @ np.diag(fp) + np.diag(f2)
                psi = np.diag(fp) @ W.T @ psi
        return cls(
            Omega=Omega0,
            psi=psi,
            sigma=sigma,
            data=(features["train"].T, train_data[1]),
            test_data=(features["test"].T, test_data[1]),
        )

    def genErrEmp(self, lamb, n=None, repeats=1):
        """
        Empirical generalization error of ridge regression with regularization parameter l and n samples

        Parameters
        ----------
        lamb : float
            The regularization parameter
        n : int
            The number of samples
        repeats : int
            The number of times to average the computation of the generalization error

        Returns
        -------
        float
            The empirical generalization error for different numbers of samples and regularization parameters
        """
        rng = np.random.default_rng()
        errors = []
        for _ in range(repeats):
            idx = rng.choice(range(len(self.y)), size=n, replace=False)
            Xt, yt = self.Phi[:, idx], self.y[idx]
            sol = tf.linalg.lstsq(Xt.T, np.array([yt]).T, l2_regularizer=lamb)
            errors.append(np.mean((self.y_test - (self.Phi_test.T @ sol)[:, 0]) ** 2))
        return np.mean(errors)

    def genErrRMT(
        self,
        lamb=None,
        n=None,
        l_bounds=None,
        maxiter=50,
    ):
        """
        Deterministic generalization error of ridge regression with regularization parameter l and n samples. If lamb is not provided, the optimal regularization parameter is found using an optimization algorithm.

        Parameters
        ----------
        lamb : float (optional)
            The regularization parameter
        n : int (default: p)
            The number of samples
        l_bounds : tuple (optional)
            The range of regularization parameters to consider
        maxiter : int (default: 50)
            The maximum number of iterations for the optimization algorithm to find the optimal regularization parameter

        Returns
        -------
        float
            The deterministic generalization error for different numbers of samples and regularization parameters
        """

        n = self.p if n is None else n

        if lamb is None and l_bounds is None:
            raise ValueError("Either provide a value for lambda, or an optimization range")
        if lamb is None:

            def opt(x):
                return self.genErrRMT(lamb=x, n=n)

            min_val = minimize_scalar(opt, bounds=l_bounds, method="bounded", options={"maxiter": maxiter})
            return {"lamb": min_val.x, "genErrRMT": min_val.fun}

        ml = self.sc.m(lamb, n=n)
        Me = 1 / (n * lamb * ml * self.sc.OmegaEig + lamb)
        TrOmMOmM = np.sum(self.sc.OmegaEig**2 * Me**2)
        psiMlM2psi = np.sum(self.UTpsi**2 * (Me + lamb * Me**2))
        return (self.sigma**2 - n * lamb * ml * psiMlM2psi) / (1 - n * (ml * lamb) ** 2 * TrOmMOmM)

    def learningCurve(self, lambdas=None, ns=None, ns_emp=None, repeats=1):
        """
        Compute the learning curve of the ridge regression model for different regularization parameters and number of samples. If lambdas is not provided, the optimal regularization parameter is found using an optimization algorithm for the deterministic generalization error, and the same lambda is used for the empirical generalization error.

        Parameters
        ----------
        lambdas : list
            The regularization parameters to consider for the deterministic generalization error
        ns : list
            The number of samples to consider for the deterministic generalization error
        ns_emp : list
            The number of samples to consider for the empirical generalization error
        repeats : int
            The number of times to average the computation of the empirical generalization error

        Returns
        -------
        pl.DataFrame
            The learning curve of the ridge regression model for different regularization parameters and number of samples
        """
        if ns is None and ns_emp is None:
            raise ValueError("Provide either ns or ns_emp")

        ns = [] if ns is None else ns
        ns_emp = [] if ns_emp is None else ns_emp
        ns = np.unique(np.concatenate([ns, ns_emp])).astype(int)

        res = []
        for n in ns:
            if not lambdas:
                new_res = self.genErrRMT(n=n, l_bounds=(1e-5, 1e5))
                new_res["n"] = n
                if n in ns_emp:
                    new_res["genErrEmp"] = self.genErrEmp(lamb=new_res["lamb"], n=n, repeats=repeats)
                res.append(new_res)
            else:
                for ll in lambdas:
                    new_res = {"n": n, "lamb": ll}
                    if n in ns:
                        new_res["genErrRMT"] = self.genErrRMT(lamb=ll, n=n)
                    if n in ns_emp:
                        new_res["genErrEmp"] = self.genErrEmp(lamb=ll, n=n, repeats=repeats)
                    res.append(new_res)
        return pl.DataFrame(res).sort("n")
