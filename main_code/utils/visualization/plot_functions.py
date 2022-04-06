import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def scatter_plot(
    x_steps,
    avg_approx_errors,
    title,
    xlabel,
    labels=None,
    save_path=None,
    ylabel="Relative error to optimal [%]",
):
    # plot stuff
    mpl.style.use("seaborn-paper")
    fig, axs = plt.subplots(1, 1, figsize=(5.66, 3.5), sharex=False, sharey=False)
    # check how mayn value to plot
    if len(avg_approx_errors) == len(x_steps):
        avg_approx_errors = [avg_approx_errors]

    for k, errors in enumerate(avg_approx_errors):
        if labels:
            label = labels[k]
        else:
            label = None
        axs.scatter(x_steps, errors, s=30, label=label)

    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.set_title(title)
    axs.set_xticks(x_steps)
    # axs.set_ylim([0, 0.02])
    axs.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)


def barplot(
    values,
    labels=None,
    xticklabels=None,
    xlabel="Inference Strategy",
    ylabel="Relative error to optimal [%]",
    title=None,
    save_path=None,
    log_plot=False,
):
    mpl.style.use("seaborn-paper")
    fig, axs = plt.subplots(1, 1, figsize=(5.66, 3.5), sharex=False, sharey=False)
    values = np.array(values)
    positions = np.arange(values.shape[0]) + 1
    if labels:
        for k, value in enumerate(values):
            axs.bar(x=positions[k], height=value, label=labels[k])
    else:
        for k, value in enumerate(values):
            axs.bar(x=positions[k], height=value)
    axs.set_xticks(positions)
    if xticklabels:
        axs.set_xticklabels(xticklabels)
    if xlabel:
        axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    if title:
        axs.set_title(title)
    if log_plot:
        axs.set_yscale("log")
    axs.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=400)
    plt.show()


def line_plot(
    df_list,
    labels,
    xvalue,
    yvalue,
    ylabel="Relative error to optimal [%]",
    xlabel="Epoch",
    xlim=None,
    ylim=None,
    suptitle=None,
    save_path=None,
    highlight_min=False,
):
    mpl.style.use("seaborn-paper")

    fig, axs = plt.subplots(1, 1, figsize=(5.66, 3.5), sharex=False, sharey=False)

    for df, label in zip(df_list, labels):
        if highlight_min:
            # calc min points
            min_idx = np.argmin(df[yvalue])
            min_x = df[xvalue].iloc[min_idx]
            min_y = np.round(df[yvalue].iloc[min_idx], 4)
            # make plot
            axs.plot(df[xvalue], df[yvalue], label=f"{label} (Minimum: {min_y}%)")
            axs.scatter(
                x=min_x,
                y=min_y,
                facecolors="none",
                linewidths=2,
                edgecolors="red",
            )
        else:
            axs.plot(df[xvalue], df[yvalue], label=f"{label}")
    # optionally add baselines
    if xlim:
        axs.set_xlim(xlim)
    if ylim:
        axs.set_ylim(ylim)

    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.grid(True, axis="both")
    axs.legend(loc="upper left")

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=400)
    plt.show()


# from finder needs rework
def plot_valid_performance(
    dfs_valid, dfs_loss, min_points, labels, subtitle, save_path
):
    mpl.style.use("seaborn-paper")

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 3.5), sharex=False, sharey=False)

    for k in range(len(dfs_valid)):
        min_point = np.round(min_points[k][0][1] * 100 - 100, 4)
        axs[0].plot(
            dfs_valid[k]["iteration"],
            dfs_valid[k]["opt_approx"] * 100 - 100,
            label=f"{labels[k]} ({min_point} %)",
        )
        axs[1].plot(
            dfs_loss[k]["iteration"],
            dfs_loss[k]["loss"],
            label=f"{labels[k]} ({min_point} %)",
        )

    for k, min_point in enumerate(min_points):
        axs[0].scatter(
            x=min_point[0][0],
            y=min_point[0][1] * 100 - 100,
            facecolors="none",
            linewidths=2,
            edgecolors="red",
        )
        # label=f'FINDER training {k+1} best model ({np.round(min_point[0][0],4)})')
    axs[0].grid(True, axis="both")
    axs[0].axhline(y=1.47, color="black", linestyle="--", label="S2V-DQN (1.47 %)")
    axs[0].axhline(
        y=0.00004, color="green", linestyle="--", label="Fu et al.(0.00004 %)"
    )
    axs[0].set_ylabel("Relative error to optimal [%]")
    axs[0].set_xlabel("Iteration (Number of batches)")
    axs[0].set_ylim([0, 25])
    axs[0].legend(loc="upper left")

    axs[1].grid(True, axis="both")
    axs[1].legend(loc="upper left")
    axs[1].set_ylabel("Training loss")
    axs[1].set_xlabel("Iteration (Number of batches)")
    # axs[1].set_ylim([0, 0.006])
    axs[1].set_yscale("log")

    plt.suptitle(subtitle)
    plt.tight_layout()

    plt.savefig(save_path, dpi=400)
