# Acknowledgement: The code is copied and developed from
# https://github.com/hildensia/bayesian_changepoint_detection/tree/master/bayesian_changepoint_detection
# and from
# https://github.com/YAU1YAN/Bayesian_HD_sparse_changepoint_online_detection/blob/master/src/multiD_online_changepoint_detection.py

"""
Multivariate Bayesian online change point detection.
"""

import numpy as np
from scipy import stats


class MultivariateBOCPD:
    def __init__(self,
                 dimensions: int,
                 lamda_gap: int = 100,
                 K: int = 1,
                 p_cp: float = 0.6,
                 n_wait: int = 5,
                 alpha: float = 0.1,
                 beta: float = 0.01,
                 mean: int = 0,
                 buffer_size: int = 2000,
                 range_threshold: int = 10):
        """
        Initialize the Multivariate Bayesian Onlinen Change Point Detection (BOCPD) model.

        Args:
            dimensions (int): The number of dimensions in the observed data.
            lamda_gap (int, optional): The hazard function parameter controlling the expected 
                                       run length of segments before a changepoint. Default is 100.
            K (int, optional): The number of largest elements to use for projection in CPD.
                               Default is 1.
            p_cp (float, optional): The probability threshold for detecting a changepoint.
                                    Default is 0.6.
            n_wait (int, optional): The minimum number of steps to wait before confirming a
                                    changepoint. Default is 5.
            alpha (float, optional): The prior shape parameter for the Student's T-distribution.
                                     Default is 0.1.
            beta (float, optional): The prior scale parameter for the Student's T-distribution.
                                    Default is 0.01.
            mean (int, optional): The prior mean of the Student's T-distribution. Default is 0.
            buffer_size (int, optional): The maximum length of the run-length matrix.
                                         Default is 2000.
            range_threshold (int, optional): The minimum distance between detected changepoints to
                                             avoid redundant detections. Default is 10.
        """
        self.dimensions = dimensions
        self.lamda_gap = lamda_gap
        self.K = K
        self.p_cp = p_cp
        self.n_wait = n_wait
        self.buffer_size = buffer_size
        self.range_threshold = range_threshold
        self.previous_cp = 0
        self.bocd_changepoints = []

        # Initialize observation likelihoods
        self.observation_likelihood_parallel = [StudentT(alpha, beta, 1, mean)
                                                for _ in range(self.dimensions)]
        self.observation_likelihood = StudentT(alpha, beta, 1, mean)

        # Initialize matrices
        self.c_prob_parallel = np.zeros((self.dimensions, buffer_size + 1),
                                        dtype=np.float32)
        self.R_parallel = [np.zeros((buffer_size + 1, buffer_size + 1), dtype=np.float32)
                           for _ in range(self.dimensions)]
        for r in self.R_parallel:
            r[0, 0] = 1

        self.R_projected = np.zeros((buffer_size + 1, buffer_size + 1), dtype=np.float32)
        self.R_projected[0, 0] = 1


    def constant_hazard(self, lam, r):
        """Calculate constant hazard function over the run-length."""
        return 1 / lam * np.ones(r.shape)


    def update(self, t, x, offset):
        """Calculate probabilities and projection. """
        detected_cp = None
        v = np.zeros(self.dimensions)
        predprobs_parallel = [obs_likelihood.pdf(obs) for obs_likelihood, obs
                              in zip(self.observation_likelihood_parallel, x)]

        for i, predprob in enumerate(predprobs_parallel):
            H_parallel = self.constant_hazard(self.lamda_gap, np.array(range(t+1)))

            self.R_parallel[i][1:t+2, t+1] = self.R_parallel[i][0:t+1, t] * predprob * (
                1 - H_parallel)
            self.R_parallel[i][0, t+1] = np.sum(
                self.R_parallel[i][0:t+1, t] * predprob * H_parallel)
            self.R_parallel[i][:, t+1] /= np.sum(self.R_parallel[i][:, t+1])

            self.observation_likelihood_parallel[i].update_theta(x[i])
            self.c_prob_parallel[i, t] = 1 - self.R_parallel[i][t, t]

        # Projection of the observed data point
        index = self.K_largest_argument(self.c_prob_parallel.T[t], self.K)
        v[index] = 1
        projected_x = np.dot(x, v)

        # CPD for the projected data
        predprobs_projected = self.observation_likelihood.pdf(projected_x)
        H_projected = self.constant_hazard(self.lamda_gap, np.array(range(t+1)))

        self.R_projected[1:t+2, t+1] = self.R_projected[0:t+1, t] * predprobs_projected * (
            1 - H_projected)
        self.R_projected[0, t+1] = np.sum(
            self.R_projected[0:t+1, t] * predprobs_projected * H_projected)
        self.R_projected[:, t+1] /= np.sum(self.R_projected[:, t+1])

        self.observation_likelihood.update_theta(projected_x)

        # Change point detection
        if self.R_projected[self.n_wait, t] >= self.p_cp and t > self.previous_cp + self.range_threshold:
            detected_cp = t + offset
            self.bocd_changepoints.append(detected_cp)
            print("Change point declared at:", detected_cp)
            self.previous_cp = t # Update last detected change point

        return detected_cp


    def K_largest_argument(self, vector, K=1):
        """Find the K largest elements in vector."""
        return np.argpartition(vector, -K)[-K:]


class StudentT: 
    def __init__(self, alpha, beta, kappa, mu):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])


    def pdf(self, data):
        """Probability density function."""
        return stats.t.pdf(x=data,
                           df=2*self.alpha, # degrees of freedom
                           loc=self.mu, # location parameter
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa))) # scale parameter


    def update_theta(self, data):
        """
        Performs update on the prior parameters, given data
        Parmeters:
            data - the datapoints to be evaluated (shape: 1 x D vector)
        """
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
