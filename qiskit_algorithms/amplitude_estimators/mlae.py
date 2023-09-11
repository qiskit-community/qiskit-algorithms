# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Maximum Likelihood Amplitude Estimation algorithm."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, List, Tuple, cast
import warnings

import numpy as np
from scipy.optimize import brute
from scipy.stats import norm, chi2

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.primitives import BaseSampler, Sampler

from .amplitude_estimator import AmplitudeEstimator, AmplitudeEstimatorResult
from .estimation_problem import EstimationProblem
from ..exceptions import AlgorithmError

MINIMIZER = Callable[[Callable[[float], float], List[Tuple[float, float]]], float]


class MaximumLikelihoodAmplitudeEstimation(AmplitudeEstimator):
    """The Maximum Likelihood Amplitude Estimation algorithm.

    This class implements the quantum amplitude estimation (QAE) algorithm without phase
    estimation, as introduced in [1]. In comparison to the original QAE algorithm [2],
    this implementation relies solely on different powers of the Grover operator and does not
    require additional evaluation qubits.
    Finally, the estimate is determined via a maximum likelihood estimation, which is why this
    class in named ``MaximumLikelihoodAmplitudeEstimation``.

    References:
        [1]: Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N. (2019).
             Amplitude Estimation without Phase Estimation.
             `arXiv:1904.10246 <https://arxiv.org/abs/1904.10246>`_.
        [2]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
             Quantum Amplitude Amplification and Estimation.
             `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(
        self,
        evaluation_schedule: list[int] | int,
        minimizer: MINIMIZER | None = None,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            evaluation_schedule: If a list, the powers applied to the Grover operator. The list
                element must be non-negative. If a non-negative integer, an exponential schedule is
                used where the highest power is 2 to the integer minus 1:
                `[id, Q^2^0, ..., Q^2^(evaluation_schedule-1)]`.
            minimizer: A minimizer used to find the minimum of the likelihood function.
                Defaults to a brute search where the number of evaluation points is determined
                according to ``evaluation_schedule``. The minimizer takes a function as first
                argument and a list of (float, float) tuples (as bounds) as second argument and
                returns a single float which is the found minimum.
            sampler: A sampler primitive to evaluate the circuits.

        Raises:
            ValueError: If the number of oracle circuits is smaller than 1.
        """

        super().__init__()

        # get parameters
        if isinstance(evaluation_schedule, int):
            if evaluation_schedule < 0:
                raise ValueError("The evaluation schedule cannot be < 0.")

            self._evaluation_schedule = [0] + [2**j for j in range(evaluation_schedule)]
        else:
            if any(value < 0 for value in evaluation_schedule):
                raise ValueError("The elements of the evaluation schedule cannot be < 0.")

            self._evaluation_schedule = evaluation_schedule

        if minimizer is None:
            # default number of evaluations is max(10^4, pi/2 * 10^3 * 2^(m))
            nevals = max(10000, int(np.pi / 2 * 1000 * 2 * self._evaluation_schedule[-1]))

            def default_minimizer(objective_fn, bounds):
                return brute(objective_fn, bounds, Ns=nevals)[0]

            self._minimizer = default_minimizer
        else:
            self._minimizer = minimizer

        self._sampler = sampler

    @property
    def sampler(self) -> BaseSampler | None:
        """Get the sampler primitive.

        Returns:
            The sampler primitive to evaluate the circuits.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler) -> None:
        """Set sampler primitive.

        Args:
            sampler: A sampler primitive to evaluate the circuits.
        """
        self._sampler = sampler

    def construct_circuits(
        self, estimation_problem: EstimationProblem, measurement: bool = False
    ) -> list[QuantumCircuit]:
        """Construct the Amplitude Estimation w/o QPE quantum circuits.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            measurement: Boolean flag to indicate if measurement should be included in the circuits.

        Returns:
            A list with the QuantumCircuit objects for the algorithm.
        """
        # keep track of the Q-oracle queries
        circuits = []

        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        q = QuantumRegister(num_qubits, "q")
        qc_0 = QuantumCircuit(q, name="qc_a")  # 0 applications of Q, only a single A operator

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            qc_0.add_register(c)

        qc_0.compose(estimation_problem.state_preparation, inplace=True)

        for k in self._evaluation_schedule:
            qc_k = qc_0.copy(name=f"qc_a_q_{k}")

            if k != 0:
                qc_k.compose(estimation_problem.grover_operator.power(k), inplace=True)

            if measurement:
                # real hardware can currently not handle operations after measurements,
                # which might happen if the circuit gets transpiled, hence we're adding
                # a safeguard-barrier
                qc_k.barrier()
                qc_k.measure(estimation_problem.objective_qubits, c[:])

            circuits += [qc_k]

        return circuits

    @staticmethod
    def compute_confidence_interval(
        result: "MaximumLikelihoodAmplitudeEstimationResult",
        alpha: float,
        kind: str = "fisher",
        apply_post_processing: bool = False,
        exact: bool = False,
    ) -> tuple[float, float]:
        """Compute the `alpha` confidence interval using the method `kind`.

        The confidence level is (1 - `alpha`) and supported kinds are 'fisher',
        'likelihood_ratio' and 'observed_fisher' with shorthand
        notations 'fi', 'lr' and 'oi', respectively.

        Args:
            result: A maximum likelihood amplitude estimation result.
            alpha: The confidence level.
            kind: The method to compute the confidence interval. Defaults to 'fisher', which
                computes the theoretical Fisher information.
            apply_post_processing: If True, apply post-processing to the confidence interval.
            exact: Whether the result comes from a statevector simulation or not

        Returns:
            The specified confidence interval.

        Raises:
            AlgorithmError: If `run()` hasn't been called yet.
            NotImplementedError: If the method `kind` is not supported.
        """
        interval: tuple[float, float] | None = None

        # if statevector simulator the estimate is exact
        if exact:
            interval = (result.estimation, result.estimation)

        elif kind in ["likelihood_ratio", "lr"]:
            interval = _likelihood_ratio_confint(result, alpha)

        elif kind in ["fisher", "fi"]:
            interval = _fisher_confint(result, alpha, observed=False)

        elif kind in ["observed_fisher", "observed_information", "oi"]:
            interval = _fisher_confint(result, alpha, observed=True)

        if interval is None:
            raise NotImplementedError(f"CI `{kind}` is not implemented.")

        if apply_post_processing:
            return result.post_processing(interval[0]), result.post_processing(interval[1])

        return interval

    # TODO: add deprecation warning for num_state_qubits
    def compute_mle(
        self,
        circuit_results: list[dict[str, int]],
        estimation_problem: EstimationProblem,
        num_state_qubits: int | None = None,  # pylint: disable=unused-argument
        return_counts: bool = False,
    ) -> float | tuple[float, list[int]]:
        """Compute the MLE via a grid-search.

        This is a stable approach if sufficient grid-points are used.

        Args:
            circuit_results: A list of circuit outcomes. Can be counts or statevectors.
            estimation_problem: The estimation problem containing the evaluation schedule and the
                number of likelihood function evaluations used to find the minimum.
            return_counts: If True, returns the good counts.
            num_state_qubits: [Deprecated]. Number of qubits to process statevector results.
        Returns:
            The MLE for the provided result object.
        """
        good_counts, all_counts = _get_counts(circuit_results, estimation_problem)

        # search range
        eps = 1e-15  # to avoid invalid value in log
        search_range = [0 + eps, np.pi / 2 - eps]

        def loglikelihood(theta):
            # loglik contains the first `it` terms of the full loglikelihood
            loglik = 0
            for i, k in enumerate(self._evaluation_schedule):
                angle = (2 * k + 1) * theta
                loglik += np.log(np.sin(angle) ** 2) * good_counts[i]
                loglik += np.log(np.cos(angle) ** 2) * (all_counts[i] - good_counts[i])
            return -loglik

        est_theta: float = self._minimizer(loglikelihood, [search_range])

        if return_counts:
            return est_theta, good_counts
        return est_theta

    def estimate(
        self, estimation_problem: EstimationProblem
    ) -> "MaximumLikelihoodAmplitudeEstimationResult":
        """Run the amplitude estimation algorithm on provided estimation problem.

        Args:
            estimation_problem: The estimation problem.

        Returns:
            An amplitude estimation results object.

        Raises:
            AlgorithmError: If `state_preparation` is not set in
                `estimation_problem`.
            AlgorithmError: Sampler job run error
        """
        if self._sampler is None:
            warnings.warn("No sampler provided, defaulting to Sampler from qiskit.primitives")
            self._sampler = Sampler()

        if estimation_problem.state_preparation is None:
            raise AlgorithmError(
                "The state_preparation property of the estimation problem must be set."
            )

        result = MaximumLikelihoodAmplitudeEstimationResult()
        result.evaluation_schedule = self._evaluation_schedule
        result.minimizer = self._minimizer
        result.post_processing = cast(Callable[[float], float], estimation_problem.post_processing)

        circuits = self.construct_circuits(estimation_problem, measurement=True)

        try:
            job = self._sampler.run(circuits)
            ret = job.result()
        except Exception as exc:
            raise AlgorithmError("The job was not completed successfully. ") from exc

        circuit_results = []
        shots = ret.metadata[0].get("shots")
        exact = True
        if shots is None:
            for quasi_dist in ret.quasi_dists:
                circuit_result = quasi_dist.binary_probabilities()
                circuit_results.append(circuit_result)
            shots = 1
        else:
            # get counts and construct MLE input
            for quasi_dist in ret.quasi_dists:
                counts = {k: round(v * shots) for k, v in quasi_dist.binary_probabilities().items()}
                circuit_results.append(counts)
            exact = False

        result.shots = shots
        result.circuit_results = circuit_results
        # run maximum likelihood estimation
        num_state_qubits = circuits[0].num_qubits - circuits[0].num_ancillas

        theta, good_counts = cast(
            Tuple[float, List[float]],
            self.compute_mle(result.circuit_results, estimation_problem, num_state_qubits, True),
        )

        # store results
        result.theta = theta
        result.good_counts = good_counts
        result.estimation = np.sin(result.theta) ** 2

        # not sure why pylint complains, this is a callable and the tests pass
        # pylint: disable=not-callable
        result.estimation_processed = result.post_processing(result.estimation)

        result.fisher_information = _compute_fisher_information(result)
        result.num_oracle_queries = result.shots * sum(k for k in result.evaluation_schedule)

        # compute and store confidence interval
        confidence_interval = self.compute_confidence_interval(
            result, alpha=0.05, kind="fisher", exact=exact
        )
        result.confidence_interval = confidence_interval
        result.confidence_interval_processed = tuple(  # type: ignore[assignment]
            estimation_problem.post_processing(value)  # type: ignore[arg-type]
            for value in confidence_interval
        )

        return result


class MaximumLikelihoodAmplitudeEstimationResult(AmplitudeEstimatorResult):
    """The ``MaximumLikelihoodAmplitudeEstimation`` result object."""

    def __init__(self) -> None:
        super().__init__()
        self._theta: float | None = None
        self._minimizer: Callable | None = None
        self._good_counts: list[float] | None = None
        self._evaluation_schedule: list[int] | None = None
        self._fisher_information: float | None = None

    @property
    def theta(self) -> float:
        r"""Return the estimate for the angle :math:`\theta`."""
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        r"""Set the estimate for the angle :math:`\theta`."""
        self._theta = value

    @property
    def minimizer(self) -> Callable:
        """Return the minimizer used for the search of the likelihood function."""
        return self._minimizer

    @minimizer.setter
    def minimizer(self, value: Callable) -> None:
        """Set the number minimizer used for the search of the likelihood function."""
        self._minimizer = value

    @property
    def good_counts(self) -> list[float]:
        """Return the percentage of good counts per circuit power."""
        return self._good_counts

    @good_counts.setter
    def good_counts(self, counts: list[float]) -> None:
        """Set the percentage of good counts per circuit power."""
        self._good_counts = counts

    @property
    def evaluation_schedule(self) -> list[int]:
        """Return the evaluation schedule for the powers of the Grover operator."""
        return self._evaluation_schedule

    @evaluation_schedule.setter
    def evaluation_schedule(self, evaluation_schedule: list[int]) -> None:
        """Set the evaluation schedule for the powers of the Grover operator."""
        self._evaluation_schedule = evaluation_schedule

    @property
    def fisher_information(self) -> float:
        """Return the Fisher information for the estimated amplitude."""
        return self._fisher_information

    @fisher_information.setter
    def fisher_information(self, value: float) -> None:
        """Set the Fisher information for the estimated amplitude."""
        self._fisher_information = value


def _safe_min(array: list[float], default: float = 0.0) -> float:
    if len(array) == 0:
        return default
    return np.min(array)


def _safe_max(
    array: list[float], default: float = (np.pi / 2)  # pylint: disable=superfluous-parens
) -> float:
    if len(array) == 0:
        return default
    return np.max(array)


def _compute_fisher_information(
    result: "MaximumLikelihoodAmplitudeEstimationResult",
    num_sum_terms: int | None = None,
    observed: bool = False,
) -> float:
    """Compute the Fisher information.

    Args:
        result: A maximum likelihood amplitude estimation result.
        num_sum_terms: The number of sum terms to be included in the calculation of the
            Fisher information. By default all values are included.
        observed: If True, compute the observed Fisher information, otherwise the theoretical
            one.

    Returns:
        The computed Fisher information, or np.inf if statevector simulation was used.

    Raises:
        KeyError: Call run() first!
    """
    a = result.estimation

    # Corresponding angle to the value a (only use real part of 'a')
    theta_a = np.arcsin(np.sqrt(np.real(a)))

    # Get the number of hits (shots_k) and one-hits (h_k)
    one_hits = result.good_counts
    all_hits = [result.shots] * len(one_hits)

    # Include all sum terms or just up to a certain term?
    evaluation_schedule = result.evaluation_schedule
    if num_sum_terms is not None:
        evaluation_schedule = evaluation_schedule[:num_sum_terms]
        # not necessary since zip goes as far as shortest list:
        # all_hits = all_hits[:num_sum_terms]
        # one_hits = one_hits[:num_sum_terms]

    # Compute the Fisher information
    fisher_information = None
    if observed:
        # Note, that the observed Fisher information is very unreliable in this algorithm!
        d_loglik = 0
        for shots_k, h_k, m_k in zip(all_hits, one_hits, evaluation_schedule):
            tan = np.tan((2 * m_k + 1) * theta_a)
            d_loglik += (2 * m_k + 1) * (h_k / tan + (shots_k - h_k) * tan)

        d_loglik /= np.sqrt(a * (1 - a))
        fisher_information = d_loglik**2 / len(all_hits)

    else:
        fisher_information = sum(
            shots_k * (2 * m_k + 1) ** 2 for shots_k, m_k in zip(all_hits, evaluation_schedule)
        )
        fisher_information /= a * (1 - a)

    return fisher_information


def _fisher_confint(
    result: MaximumLikelihoodAmplitudeEstimationResult, alpha: float = 0.05, observed: bool = False
) -> tuple[float, float]:
    """Compute the `alpha` confidence interval based on the Fisher information.

    Args:
        result: A maximum likelihood amplitude estimation results object.
        alpha: The level of the confidence interval (must be <= 0.5), default to 0.05.
        observed: If True, use observed Fisher information.

    Returns:
        float: The alpha confidence interval based on the Fisher information
    Raises:
        AssertionError: Call run() first!
    """
    # Get the (observed) Fisher information
    fisher_information = None
    try:
        fisher_information = result.fisher_information
    except KeyError as ex:
        raise AssertionError("Call run() first!") from ex

    if observed:
        fisher_information = _compute_fisher_information(result, observed=True)

    normal_quantile = norm.ppf(1 - alpha / 2)
    confint = np.real(result.estimation) + normal_quantile / np.sqrt(fisher_information) * np.array(
        [-1, 1]
    )
    return result.post_processing(confint[0]), result.post_processing(confint[1])


def _likelihood_ratio_confint(
    result: MaximumLikelihoodAmplitudeEstimationResult,
    alpha: float = 0.05,
    nevals: int | None = None,
) -> tuple[float, float]:
    """Compute the likelihood-ratio confidence interval.

    Args:
        result: A maximum likelihood amplitude estimation results object.
        alpha: The level of the confidence interval (< 0.5), defaults to 0.05.
        nevals: The number of evaluations to find the intersection with the loglikelihood
            function. Defaults to an adaptive value based on the maximal power of Q.

    Returns:
        The alpha-likelihood-ratio confidence interval.
    """
    if nevals is None:
        nevals = max(10000, int(np.pi / 2 * 1000 * 2 * result.evaluation_schedule[-1]))

    def loglikelihood(theta, one_counts, all_counts):
        loglik = 0
        for i, k in enumerate(result.evaluation_schedule):
            loglik += np.log(np.sin((2 * k + 1) * theta) ** 2) * one_counts[i]
            loglik += np.log(np.cos((2 * k + 1) * theta) ** 2) * (all_counts[i] - one_counts[i])
        return loglik

    one_counts = result.good_counts
    all_counts = [result.shots] * len(one_counts)

    eps = 1e-15  # to avoid invalid value in log
    thetas = np.linspace(0 + eps, np.pi / 2 - eps, nevals)
    values = np.zeros(len(thetas))
    for i, theta in enumerate(thetas):
        values[i] = loglikelihood(theta, one_counts, all_counts)

    loglik_mle = loglikelihood(result.theta, one_counts, all_counts)
    chi2_quantile = chi2.ppf(1 - alpha, df=1)
    thres = loglik_mle - chi2_quantile / 2

    # the (outer) LR confidence interval
    above_thres = thetas[values >= thres]

    # it might happen that the `above_thres` array is empty,
    # to still provide a valid result use safe_min/max which
    # then yield [0, pi/2]
    confint = [_safe_min(above_thres, default=0), _safe_max(above_thres, default=np.pi / 2)]
    mapped_confint = cast(
        Tuple[float, float], tuple(result.post_processing(np.sin(bound) ** 2) for bound in confint)
    )

    return mapped_confint


def _get_counts(
    circuit_results: Sequence[dict[str, int]], estimation_problem: EstimationProblem
) -> tuple[list[int], list[int]]:
    """Get the good and total counts.

    Returns:
        A pair of two lists, ([1-counts per experiment], [shots per experiment]).

    Raises:
        AlgorithmError: If self.run() has not been called yet.
    """
    one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
    all_hits = []
    for counts in circuit_results:
        all_hits.append(sum(counts.values()))
        one_hits.append(
            sum(
                count
                for bitstr, count in counts.items()
                if estimation_problem.is_good_state(bitstr)
            )
        )

    return one_hits, all_hits
