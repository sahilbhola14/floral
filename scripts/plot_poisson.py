import numpy as np
import matplotlib.pyplot as plt
import plot_utils as utils

plt.style.use("journal.mplstyle")

# Parameters
n_plot = 5

# Data files
non_corrFlow = np.load("poisson_non_corrFlow_prediction.npz", allow_pickle=True)
corrFlow = np.load("poisson_corrFlow_prediction.npz", allow_pickle=True)


def get_field(data, field_type="true"):
    assert field_type in [
        "true",
        "low",
        "pred",
    ], "field_type must be 'true', 'low', or 'pred'"
    field = data["field"].item()
    return field[field_type]


def get_qoi(data, qoi_type="true"):
    assert qoi_type in [
        "true",
        "low",
        "pred",
    ], "qoi_type must be 'true', 'low', or 'pred'"
    qoi = data["qoi"].item()
    return qoi[qoi_type]


def get_residual(data, field_type="true"):
    assert field_type in ["true", "pred"], "field_type must be 'true' or 'pred'"
    residual = data["residual"].item()
    return residual[field_type]


def get_domain(data, domain_type="high"):
    assert domain_type in ["high", "low"], "domain_type must be 'high' or 'low'"
    domain = data["domain"].item()
    return np.array(domain[domain_type]).ravel()


def compute_Mean_CRPS(prediction, target):
    """Compute the Mean Continuous Ranked Probability Score (CRPS)"""
    crps = []
    for isample in range(len(prediction)):
        p = np.asarray(prediction[isample])
        t = float(target[isample])

        term1 = np.mean(np.abs(p - t))
        term2 = 0.5 * np.mean(np.abs(p[:, None] - p[None, :]))

        crps.append(term1 - term2)

    return np.mean(crps)


if __name__ == "__main__":
    # Field
    true_field = get_field(corrFlow, "true")
    low_field = get_field(corrFlow, "low")
    pred_field_corrFlow = get_field(corrFlow, "pred")
    pred_field_non_corrFlow = get_field(non_corrFlow, "pred")

    # Domain
    domain_high = get_domain(corrFlow, "high")
    domain_low = get_domain(corrFlow, "low")

    # QoI
    true_qoi = get_qoi(corrFlow, "true")
    low_qoi = get_qoi(corrFlow, "low")
    pred_qoi_corrFlow = get_qoi(corrFlow, "pred")
    pred_qoi_non_corrFlow = get_qoi(non_corrFlow, "pred")

    # Residual
    true_residual = get_residual(corrFlow, "true")
    residual_corrFlow = get_residual(corrFlow, "pred")
    residual_non_corrFlow = get_residual(non_corrFlow, "pred")

    # CRPS
    crps_corrFlow = compute_Mean_CRPS(pred_qoi_corrFlow, true_qoi).item()
    crps_non_corrFlow = compute_Mean_CRPS(pred_qoi_non_corrFlow, true_qoi).item()

    # Plotting
    fig, axs = plt.subplots(
        2, n_plot, figsize=(n_plot * 2.5, 5), sharex=True, sharey=True
    )
    axs = axs.ravel()
    for ii in range(n_plot):
        # Field plots
        axs[ii].plot(domain_high, true_field[ii], "k", label="True")
        # axs[ii].plot(
        #     domain_low,
        #     low_field[ii],
        #     "m",
        #     label="Low Fideltiy",
        #     linestyle="--",
        #     marker="s",
        #     markersize=4,
        # )

        mean_corrFlow_prediciton = np.mean(pred_field_corrFlow[ii], axis=0)
        mean_non_corrFlow_prediciton = np.mean(pred_field_non_corrFlow[ii], axis=0)

        std_corrFlow_prediciton = np.std(pred_field_corrFlow[ii], axis=0)
        std_non_corrFlow_prediciton = np.std(pred_field_non_corrFlow[ii], axis=0)

        axs[ii].plot(
            domain_high,
            mean_non_corrFlow_prediciton,
            "g",
            label="High-fidelity",
            linestyle="--",
        )
        axs[ii].fill_between(
            domain_high,
            mean_non_corrFlow_prediciton - 3.0 * std_non_corrFlow_prediciton,
            mean_non_corrFlow_prediciton + 3.0 * std_non_corrFlow_prediciton,
            color="g",
            alpha=0.2,
        )

        axs[ii].plot(
            domain_high, mean_corrFlow_prediciton, "r", label="CorrFlow", linestyle="-."
        )
        axs[ii].fill_between(
            domain_high,
            mean_corrFlow_prediciton - 3.0 * std_corrFlow_prediciton,
            mean_corrFlow_prediciton + 3.0 * std_corrFlow_prediciton,
            color="r",
            alpha=0.2,
        )

        axs[ii].set_xlabel(r"$x$")
        axs[ii].set_ylabel(r"$u(x, f(x))$")
        axs[ii].label_outer()
        if ii == 0:
            axs[ii].legend(fontsize=10)

        # Residual plots
        axs[ii + n_plot].plot(domain_high, true_residual[ii], "k", label="True")
        mean_corrFlow_residual = np.mean(residual_corrFlow[ii], axis=0)
        mean_non_corrFlow_residual = np.mean(residual_non_corrFlow[ii], axis=0)
        std_corrFlow_residual = np.std(residual_corrFlow[ii], axis=0)
        std_non_corrFlow_residual = np.std(residual_non_corrFlow[ii], axis=0)

        axs[ii + n_plot].plot(
            domain_high,
            mean_non_corrFlow_residual,
            "g",
            label="High-fidelity",
            linestyle="--",
        )
        axs[ii + n_plot].fill_between(
            domain_high,
            mean_non_corrFlow_residual - 2.0 * std_non_corrFlow_residual,
            mean_non_corrFlow_residual + 2.0 * std_non_corrFlow_residual,
            color="g",
            alpha=0.2,
        )
        axs[ii + n_plot].plot(
            domain_high, mean_corrFlow_residual, "r", label="CorrFlow", linestyle="-."
        )
        axs[ii + n_plot].fill_between(
            domain_high,
            mean_corrFlow_residual - 2.0 * std_corrFlow_residual,
            mean_corrFlow_residual + 2.0 * std_corrFlow_residual,
            color="r",
            alpha=0.2,
        )
        axs[ii + n_plot].set_xlabel(r"$x$")
        axs[ii + n_plot].set_ylabel(r"Residual")
        axs[ii + n_plot].label_outer()

    # Super title
    sup_title = (
        r"$\mathrm{E}[\mathrm{CRPS}_{\mathrm{corrFlow}}(q)] = {%s}$, "
        r"$\mathrm{E}[\mathrm{CRPS}_{\mathrm{High\text{-}fidelity}}(q)] = {%s}$"
        % (utils.to_latex_sci(crps_corrFlow), utils.to_latex_sci(crps_non_corrFlow))
    )

    fig.suptitle(sup_title, fontsize=12)

    plt.tight_layout()
    plt.savefig("poisson.png", dpi=300, bbox_inches="tight")
    plt.close()
