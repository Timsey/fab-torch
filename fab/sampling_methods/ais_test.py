from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

from fab.sampling_methods import AnnealedImportanceSampler, Metropolis, HamiltonianMonteCarlo
from fab.utils.logging import ListLogger
from fab.target_distributions import TargetDistribution
from fab.target_distributions.gmm import GMM
from fab.wrappers.torch import WrappedTorchDist
from fab.utils.plotting import plot_history
from fab.sampling_methods.base import resample

def setup_ais(dim: int = 2,
            n_ais_intermediate_distributions: int = 40,
            seed: int = 0,
            transition_operator_type: str = "hmc",
            spacing: str = "geometric",
            p_sq_over_q_target: bool = False) -> \
        Tuple[AnnealedImportanceSampler, TargetDistribution]:
    # set up key objects
    torch.manual_seed(seed)
    target = GMM(dim=dim, n_mixes=4, loc_scaling=8)
    base_dist = WrappedTorchDist(torch.distributions.MultivariateNormal(loc=torch.zeros(dim),
                                                                 scale_tril=3*torch.eye(dim)))
    # setup transition operator
    if transition_operator_type == "hmc":
        transition_operator = HamiltonianMonteCarlo(
            n_ais_intermediate_distributions=n_ais_intermediate_distributions,
            dim=dim,
            base_log_prob=base_dist.log_prob,
            target_log_prob=target.log_prob,
            p_sq_over_q_target=p_sq_over_q_target,
            n_outer=5,
            epsilon=1.0,
            L=5,
        )
    elif transition_operator_type == "metropolis":
        transition_operator = Metropolis(n_transitions=n_ais_intermediate_distributions,
                                         n_updates=5)
    else:
        raise NotImplementedError
    ais = AnnealedImportanceSampler(base_distribution=base_dist,
                                    target_log_prob=target.log_prob,
                                    transition_operator=transition_operator,
                                    n_intermediate_distributions=n_ais_intermediate_distributions,
                                    distribution_spacing_type=spacing,
                                    p_sq_over_q_target=p_sq_over_q_target
                                    )
    return ais, target


def test_ais__eval_batch(
        outer_batch_size: int = 100,
        inner_batch_size: int = 100,
        dim: int = 2,
        n_ais_intermediate_distributions: int = 40,
        seed: int = 0,
        transition_operator_type: str = "hmc",
             ):
    ais, _ = setup_ais(dim=dim, n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                    seed=seed, transition_operator_type=transition_operator_type)
    base_samples, base_log_w, ais_samples, ais_log_w = ais.generate_eval_data(outer_batch_size,
                                                                              inner_batch_size)
    assert base_samples.shape == (outer_batch_size, dim)
    assert ais_samples.shape == (outer_batch_size, dim)
    assert base_log_w.shape == (outer_batch_size,)
    assert ais_log_w.shape == (outer_batch_size, )



def test_ais__overall(dim: int = 2,
            n_ais_intermediate_distributions: int = 40,
            n_iterations: int = 40,
            batch_size: int = 1000,
            seed: int = 0,
            transition_operator_type: str = "hmc",
             ) -> None:
    ais, target = setup_ais(dim=dim,
                          n_ais_intermediate_distributions=n_ais_intermediate_distributions,
                    seed=seed, transition_operator_type=transition_operator_type)
    logger = ListLogger()

    # set up plotting
    n_plots = 4
    n_plots_total = n_plots + 2
    fig, axs = plt.subplots(n_plots_total, 2, figsize=(2*3, n_plots_total*3), sharex=True, sharey=True)
    plot_number_iterator = iter(range(n_plots_total))

    # plot base and target
    true_samples = target.sample((batch_size,)).cpu().detach()
    plot_index = next(plot_number_iterator)
    axs[plot_index, 0].plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    axs[plot_index, 0].set_title("target samples")
    axs[plot_index, 1].plot(true_samples[:, 0], true_samples[:, 1], "o", alpha=0.5)
    axs[plot_index, 1].set_title("target samples")

    x_base, log_q_base = ais.base_distribution.sample_and_log_prob((batch_size,))
    x_base, log_q_base = x_base.cpu().detach(), log_q_base.cpu().detach()
    plot_index = next(plot_number_iterator)
    plotting_iterations = list(np.linspace(0, n_iterations-1, n_plots, dtype="int"))
    axs[plot_index, 0].plot(x_base[:, 0], x_base[:, 1], "o", alpha=0.5)
    axs[plot_index, 0].set_title("base samples")

    # estimate performance metrics over base distribution
    log_w = ais.target_log_prob(x_base) - log_q_base
    performance_metrics = target.performance_metrics(x_base, log_w)
    print(f"Performance metrics over base distribution {performance_metrics}")

    # plot resampled base poits
    x_base_resampled = resample(x_base, log_w)
    axs[plot_index, 1].plot(x_base_resampled[:, 0], x_base_resampled[:, 1], "o", alpha=0.5)
    axs[plot_index, 1].set_title("base samples resampled")


    # run test
    for i in range(n_iterations):
        points, log_w = ais.sample_and_log_weights(batch_size=batch_size)
        performance_metrics = target.performance_metrics(points.x, log_w)
        logging_info = ais.get_logging_info()
        logging_info.update(performance_metrics)
        logger.write(logging_info)

        # Plotting progress as the transition operator gets tuned.
        if i in plotting_iterations:
            x = points.x.cpu().detach()
            plot_index = next(plot_number_iterator)
            axs[plot_index, 0].plot(x[:, 0], x[:, 1], "o", alpha=0.5)
            axs[plot_index, 0].set_title(f"samples, iteration {i}")
            fig.show()
            resampled_points = resample(points, log_w).x
            axs[plot_index, 1].plot(resampled_points[:, 0], resampled_points[:, 1],
                                    "o", alpha=0.5)
            axs[plot_index, 1].set_title(f"resampled AIS points, iteration {i}")


    plt.tight_layout()
    fig.show()

    plot_history(logger.history)
    plt.show()


if __name__ == '__main__':
    test_ais__overall(n_ais_intermediate_distributions=4)


