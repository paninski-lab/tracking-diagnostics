"""Helper and plotting functions for paper figures."""

import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec
import matplotlib.ticker as mticker
import numpy as np
import os
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns


labels_fontsize = 12


# -------------------------------------------------------------------------------------------------
# these functions produce plots in a single axis
# -------------------------------------------------------------------------------------------------
def get_frame_examples(
        df_labeled_metrics, df_labeled_preds, df_ground_truth, project_dir,
        n_max_frames=4, conf_thresh=0.95, pix_err_thresh=10,
        pca_type='pca_singleview_error', rng_seed_data_pt='0', split_set='test', train_frames='75',
        model_type='baseline'):

    cols_to_drop = [
        'set', 'metric', 'distribution', 'model_path', 'rng_seed_data_pt', 'train_frames',
        'model_type', 'mean',
    ]
    # subselect metrics
    mask = ((df_labeled_metrics.set == split_set)
            & (df_labeled_metrics.train_frames == train_frames)
            & (df_labeled_metrics.distribution == 'OOD')
            & (df_labeled_metrics.model_type == model_type)
            & (df_labeled_metrics.rng_seed_data_pt == rng_seed_data_pt)
            )
    df_test = df_labeled_metrics[mask]
    # just look at means across keypoints for each metric; "sample" will shuffle the df
    np.random.seed(0)
    df_means = df_test.pivot(columns='metric', values='mean').sample(frac=1)
    # retain frames that have high confidence and high pixel error
    df_selected = df_means[
        (df_means.confidence > conf_thresh) & (df_means.pixel_error > pix_err_thresh)]
    # get df for a single frame; one row for each metric, one col for each keypoint
    results = []
    for idx, index in enumerate(df_selected.index):
        if len(results) == n_max_frames:
            break
        df_image = df_test.loc[index]
        # now look at individual keypoints
        # retain keypoints with high likelihood
        kp_filter1 = df_image[df_image.metric == 'confidence'].drop(
            columns=cols_to_drop) > conf_thresh
        # retain keypoints with labels
        kp_filter2 = df_image[df_image.metric == 'pixel_error'].drop(
            columns=cols_to_drop) > pix_err_thresh
        kp_filter = kp_filter1 & kp_filter2
        # get keypoint that survives these filters with highest pca singleview error
        kp_name = df_image[df_image.metric == pca_type].drop(columns=cols_to_drop)[kp_filter].idxmax(axis=1)[0]
        if not isinstance(kp_name, str):
            # this happens if we filtered out all columns
            continue
        frame_name = df_image.index[0]
        kp_conf = df_image[df_image.metric == 'confidence'][kp_name][0]
        kp_pix_err = df_image[df_image.metric == 'pixel_error'][kp_name][0]
        kp_pca_err = df_image[df_image.metric == pca_type][kp_name][0]
        if kp_pca_err < pix_err_thresh:
            continue
        # get predictions
        mask_ = ((df_labeled_preds.set == split_set)
                 & (df_labeled_preds.train_frames == train_frames)
                 & (df_labeled_preds.distribution == 'OOD')
                 & (df_labeled_preds.model_type == model_type)
                 & (df_labeled_preds.rng_seed_data_pt == rng_seed_data_pt)
                 )
        df_coords = df_labeled_preds[mask_ & (df_labeled_preds.index == frame_name)][kp_name]
        file_name = os.path.join(project_dir, frame_name)
        image = Image.open(file_name).convert("RGB")
        results.append({
            'frame': image,
            'x_pred': df_coords['x'][0],
            'y_pred': df_coords['y'][0],
            'x_true': df_ground_truth.loc[frame_name, kp_name]['x'],
            'y_true': df_ground_truth.loc[frame_name, kp_name]['y'],
            'kp_name': kp_name,
            'confidence': kp_conf,
            'pixel_error': kp_pix_err,
            'pca_error': kp_pca_err,
        })
    return results


# -------------------------------------------------------------------------------------------------
# these functions produce plots in a single axis
# -------------------------------------------------------------------------------------------------
def pearsonr_ci(x, y, alpha=0.05):
    """calculate Pearson correlation along with the confidence interval using scipy and numpy

    from https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/

    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default

    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals

    """
    r, p = stats.pearsonr(x, y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


def plot_scatters(
        df, metric_names, train_frames, split_set, distribution, model_types, keypoint, ax,
        add_diagonal=False, add_trendline=False, markersize=None, alpha=0.25, scale='linear',
        color='k', trendline_kwargs={},
):
    """Plot scatters using matplotlib"""

    if scale == 'linear':
        scale_x = 'linear'
        scale_y = 'linear'
    else:
        scale_x = 'log'
        scale_y = 'log'

    # symbols = ['.', '+', '^', 's', 'o']
    symbols = ['.', '.', '.', '.', '.']
    mask_0 = get_scatter_mask(
        df=df, metric_name=metric_names[0], train_frames=train_frames, split_set=split_set,
        distribution=distribution, model_type=model_types[0])
    mask_1 = get_scatter_mask(
        df=df, metric_name=metric_names[1], train_frames=train_frames, split_set=split_set,
        distribution=distribution, model_type=model_types[1])
    df_xs = df[mask_0][keypoint]
    df_ys = df[mask_1][keypoint]
    assert np.all(df_xs.index == df_ys.index)
    xs = df_xs.to_numpy()
    ys = df_ys.to_numpy()
    if np.nanmax(ys) <= 1.0:
        scale_y = 'linear'
    if np.nanmax(xs) <= 1.0:
        scale_x = 'linear'
    if scale_x == 'log':
        xs = np.log10(xs)
    if scale_y == 'log':
        ys = np.log10(ys)
    rng_seed = df[mask_0].rng_seed_data_pt.to_numpy()
    # mn = np.nanmin([np.nanpercentile(xs, 1), np.nanpercentile(ys, 1)])
    # mx = np.nanmax([np.nanpercentile(xs, 99), np.nanpercentile(ys, 99)])
    mn = np.nanmin([np.nanmin(xs), np.nanmin(ys)])
    mx = np.nanmax([np.nanmax(xs), np.nanmax(ys)])
    if color == 'by_video':
        video_names = df[mask_0].video_name.to_numpy()
        for j, r in enumerate(np.unique(rng_seed)):
            ax.set_prop_cycle(None)
            for i, video_name in enumerate(np.unique(video_names)):
                ax.scatter(
                    xs[(rng_seed == r) & (video_names == video_name)],
                    ys[(rng_seed == r) & (video_names == video_name)],
                    marker=symbols[j], s=markersize, alpha=alpha, edgecolors='none',
                    label='RNG seed %s, video %s' % (r, video_name))
    else:
        for j, r in enumerate(np.unique(rng_seed)):
            ax.scatter(
                xs[rng_seed == r], ys[rng_seed == r], marker=symbols[j], color=color,
                s=markersize, alpha=alpha, label='RNG seed %s' % r)

    # span = mx - mn
    # m1 = mn - 0.05 * span
    # m2 = mx + 0.05 * span
    # ax.set_xlim(m1, m2)
    # ax.set_ylim(m1, m2)
    # print(m1)
    # print(m2)
    # ax.set_xlim(-0.3, 2)
    # ax.set_ylim(-0.3, 2)

    # https://stackoverflow.com/questions/63723514/userwarning-fixedformatter-should-only-be-used-together-with-fixedlocator
    label_format = '{:,.1f}'
    if scale_x == 'log':
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels([label_format.format(10 ** x) for x in ticks_loc])
    if scale_y == 'log':
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ticks_loc = ax.get_yticks().tolist()
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels([label_format.format(10 ** y) for y in ticks_loc])

    ret_vals = None
    if add_diagonal:
        span = mx - mn
        if not np.isnan(span):
            ax.plot([mn, mx], [mn, mx], 'k')
            if scale == 'linear':
                ax.set_xlim(mn - 0.05 * span, mx + 0.05 * span)
                ax.set_ylim(mn - 0.05 * span, mx + 0.05 * span)

    if add_trendline:
        nan_idxs = np.isnan(xs) | np.isnan(ys)
        xs_nonan = xs[~nan_idxs]
        ys_nonan = ys[~nan_idxs]
        zs = np.polyfit(xs_nonan, ys_nonan, 1)
        p = np.poly1d(zs)
        # r_val, p_val = stats.pearsonr(xs_nonan, ys_nonan)
        r_val, p_val, lo, hi = pearsonr_ci(xs_nonan, ys_nonan)
        xs_sorted = np.sort(xs_nonan)
        # ax.plot(xs_sorted, p(xs_sorted), '--r')
        ax.plot(xs_sorted, p(xs_sorted), **trendline_kwargs)
        ret_vals = r_val, p_val, lo, hi

    return ret_vals


def get_scatter_mask(
        df, metric_name, train_frames, model_type, split_set=None, distribution=None,
        rng_seed=None):
    mask = ((df.metric == metric_name)
            & (df.train_frames == train_frames)
            & (df.model_type == model_type))
    if split_set is not None:
        mask = mask & (df.set == split_set)
    if distribution is not None:
        mask = mask & (df.distribution == distribution)
    if rng_seed is not None:
        mask = mask & (df.rng_seed_data_pt == rng_seed)
    return mask


def get_trace_mask(df, video_name, train_frames, model_type, rng_seed, metric_name=None):
    mask = ((df.train_frames == train_frames)
            & (df.rng_seed_data_pt == rng_seed)
            & (df.model_type == model_type)
            & (df.video_name == video_name))
    if metric_name is not None:
        mask = mask & (df.metric == metric_name)
    return mask


def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold', fontsize=14)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')


# -------------------------------------------------------------------------------------------------
# these functions produce entire figures
# -------------------------------------------------------------------------------------------------
def plot_traces_and_metrics(
        df_video_metrics, df_video_preds, models_to_compare, keypoint, vid_name, train_frames,
        rng_seed, time_window, display_plot=True, save_file=None):

    colors = px.colors.qualitative.Plotly

    rows = 3
    row_heights = [2, 2, 0.75]
    metrics = df_video_metrics.metric.unique()
    if "temporal_norm" in metrics:
        rows += 1
        row_heights.insert(0, 0.75)
    if "pca_multiview_error" in metrics:
        rows += 1
        row_heights.insert(0, 0.75)
    if "pca_singleview_error" in metrics:
        rows += 1
        row_heights.insert(0, 0.75)

    fig_traces = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        x_title="Frame number",
        row_heights=row_heights,
        vertical_spacing=0.03,
    )

    yaxis_labels = {}
    row = 1

    # plot temporal norms
    if "temporal_norm" in metrics:
        for c, model_type in enumerate(models_to_compare):
            mask = get_trace_mask(
                df_video_metrics, video_name=vid_name, metric_name="temporal_norm",
                train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
            fig_traces.add_trace(
                go.Scatter(
                    name=model_type,
                    x=np.arange(time_window[0], time_window[1]),
                    y=df_video_metrics[mask][keypoint][slice(*time_window)],
                    mode='lines',
                    line=dict(color=colors[c]),
                    showlegend=False,
                ),
                row=row, col=1
            )
        yaxis_labels['yaxis%i' % row] = "temporal<br>norm"
        row += 1

    # plot pca multiview reprojection errors
    if "pca_multiview_error" in metrics:
        for c, model_type in enumerate(models_to_compare):
            mask = get_trace_mask(
                df_video_metrics, video_name=vid_name, metric_name="pca_multiview_error",
                train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
            fig_traces.add_trace(
                go.Scatter(
                    name=model_type,
                    x=np.arange(time_window[0], time_window[1]),
                    y=df_video_metrics[mask][keypoint][slice(*time_window)],
                    mode='lines',
                    line=dict(color=colors[c]),
                    showlegend=False,
                ),
                row=row, col=1
            )
        yaxis_labels['yaxis%i' % row] = "pca multi<br>error"
        row += 1

    # plot pca singleview reprojection errors
    if "pca_singleview_error" in metrics:
        for c, model_type in enumerate(models_to_compare):
            mask = get_trace_mask(
                df_video_metrics, video_name=vid_name, metric_name="pca_singleview_error",
                train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
            fig_traces.add_trace(
                go.Scatter(
                    name=model_type,
                    x=np.arange(time_window[0], time_window[1]),
                    y=df_video_metrics[mask][keypoint][slice(*time_window)],
                    mode='lines',
                    line=dict(color=colors[c]),
                    showlegend=False,
                ),
                row=row, col=1
            )
        yaxis_labels['yaxis%i' % row] = "pca single<br>error"
        row += 1

    # plot traces
    for coord in ["x", "y"]:
        for c, model_type in enumerate(models_to_compare):
            mask = get_trace_mask(
                df_video_preds, video_name=vid_name,
                train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
            fig_traces.add_trace(
                go.Scatter(
                    name=model_type,
                    x=np.arange(time_window[0], time_window[1]),
                    y=df_video_preds[mask].loc[:, (keypoint, coord)][slice(*time_window)],
                    mode='lines',
                    line=dict(color=colors[c]),
                    showlegend=False if coord == "x" else True,
                ),
                row=row, col=1
            )
        yaxis_labels['yaxis%i' % row] = "%s coordinate" % coord
        row += 1

    # plot likelihoods
    for c, model_type in enumerate(models_to_compare):
        mask = get_trace_mask(
            df_video_preds, video_name=vid_name,
            train_frames=train_frames, model_type=model_type, rng_seed=rng_seed)
        fig_traces.add_trace(
            go.Scatter(
                name=model_type,
                x=np.arange(time_window[0], time_window[1]),
                y=df_video_preds[mask].loc[:, (keypoint, "likelihood")][slice(*time_window)],
                mode='lines',
                line=dict(color=colors[c]),
                showlegend=False,
            ),
            row=row, col=1
        )
    yaxis_labels['yaxis%i' % row] = "confidence"
    row += 1

    for k, v in yaxis_labels.items():
        fig_traces["layout"][k]["title"] = v
    fig_traces.update_layout(
        width=800, height=np.sum(row_heights) * 125,
        title_text="Timeseries of %s" % keypoint
    )

    if save_file is not None:
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        fig_traces.write_image(save_file)

    if display_plot:
        fig_traces.show()


def plot_metric_bars_and_scatters_labeled(
        df_labeled_metrics, models_to_compare, keypoint, train_frames, split_set, pix_thresh=0,
        title=None, display_plot=True, save_file=None):
    """First row is barplots of metrics across all models, second row is scatters for two models"""

    plots = {
        'pixel_error': 'Pixel error',
        'pca_singleview_error': 'Pose PCA',
    }
    if 'pca_multiview_error' in df_labeled_metrics.metric.unique():
        plots['pca_multiview_error'] = 'Multi-view PCA'

    # row 1: barplots of metrics for all models
    # row 2: scatterplots of metrics on OOD data for 2 models
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows, len(plots), figsize=(4 * len(plots), 3.5 * n_rows + 0.5), squeeze=False)

    for i, (metric_name, ax_title) in enumerate(plots.items()):
        mask = ((df_labeled_metrics.set == split_set)
                & (df_labeled_metrics.metric == metric_name)
                & (df_labeled_metrics.train_frames == train_frames))

        # row 1: barplots of metrics for all models
        df_tmp = df_labeled_metrics[mask]
        sns.barplot(
            x='distribution', y=keypoint, hue='model_type',
            hue_order=['baseline', 'context', 'semi-super', 'semi-super context'],
            data=df_tmp[df_tmp[keypoint] > pix_thresh],
            ax=axes[0][i],
            #         showfliers=False,
        )
        axes[0][i].set_title(ax_title)
        axes[0][i].set_ylabel('Error (pix)', fontsize=labels_fontsize)
        axes[0][i].set_xlabel('Distribution', fontsize=labels_fontsize)
        #     axes[0][i].set_yscale('log')
        sns.move_legend(axes[0][i], "lower left")
        if i != 0:
            axes[0][i].get_legend().remove()

        # row 2: scatterplots of metrics on OOD data for 2 models
        plot_scatters(
            df=df_labeled_metrics, metric_names=[metric_name, metric_name],
            train_frames=train_frames, split_set=split_set, distribution='OOD',
            model_types=models_to_compare, keypoint=keypoint, ax=axes[1][i], add_diagonal=True,
            markersize=5, scale='log')
        axes[1][i].set_title('%s (OOD data)' % ax_title)
        axes[1][i].set_xlabel(
            '%s model error (pix)' % (models_to_compare[0].capitalize()), fontsize=labels_fontsize)
        axes[1][i].set_ylabel(
            '%s model error (pix)' % (models_to_compare[1].capitalize()), fontsize=labels_fontsize)
        if i == 0:
            axes[1][i].legend()

    if title is not None:
        plt.subplots_adjust(top=0.95)
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_file is not None:
        savefig(save_file)
    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_metric_vs_pixerror_scatters(
        df_labeled_metrics, models_to_compare, keypoint, train_frames, split_set,
        title=None, display_plot=True, save_file=None):
    """Metric vs pixel error for multiple models."""

    plots = {
        'confidence': 'Confidence',
        'pca_singleview_error': 'Pose PCA (pix)',
    }
    if 'pca_multiview_error' in df_labeled_metrics.metric.unique():
        plots['pca_multiview_error'] = 'Multi-view PCA (pix)'

    n_cols = len(plots)
    n_rows = len(models_to_compare)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows + 0.5), squeeze=False)
    grid = plt.GridSpec(n_rows, n_cols)

    for j, model_type in enumerate(models_to_compare):
        create_subtitle(fig, grid[j, ::], model_type.capitalize())
        for i, (metric_name, ax_title) in enumerate(plots.items()):
            r_val, p_val, _, _ = plot_scatters(
                df=df_labeled_metrics, metric_names=['pixel_error', metric_name],
                train_frames=train_frames, split_set=split_set, distribution='OOD',
                model_types=[model_type, model_type], keypoint=keypoint, ax=axes[j][i],
                add_trendline=True, markersize=5, scale='log')
            axes[j][i].set_title('r=%1.2f [p=%1.3f]' % (r_val, p_val))
            axes[j][i].set_xlabel('Pixel error', fontsize=labels_fontsize)
            axes[j][i].set_ylabel('%s' % ax_title, fontsize=labels_fontsize)
            if (i == 0) and (j == 0):
                axes[j][i].legend()

    if title is not None:
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_file is not None:
        savefig(save_file)
    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_metric_vs_metric_scatters(
        df_labeled_metrics, models_to_compare, keypoint, train_frames, split_set,
        title=None, display_plot=True, save_file=None):
    """Metric vs metric for multiple models."""

    plots = {
        'pca_singleview_error': 'Pose PCA (pix)',
    }
    if 'pca_multiview_error' in df_labeled_metrics.metric.unique():
        plots['pca_multiview_error'] = 'Multi-view PCA (pix)'

    n_cols = len(plots) if 'pca_multiview_error' not in plots else len(plots) + 1
    n_rows = len(models_to_compare)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows + 0.5), squeeze=False)

    grid = plt.GridSpec(n_rows, n_cols)

    for j, model_type in enumerate(models_to_compare):
        create_subtitle(fig, grid[j, ::], model_type.capitalize())
        for i, (metric_name, ax_title) in enumerate(plots.items()):
            r_val, p_val, _, _ = plot_scatters(
                df=df_labeled_metrics, metric_names=['confidence', metric_name],
                train_frames=train_frames, split_set=split_set, distribution='OOD',
                model_types=[model_type, model_type], keypoint=keypoint, ax=axes[j][i],
                add_trendline=True, markersize=5, scale='linear')
            axes[j][i].set_title('r=%1.2f [p=%1.3f]' % (r_val, p_val))
            axes[j][i].set_xlabel('Confidence', fontsize=labels_fontsize)
            axes[j][i].set_ylabel('%s' % ax_title, fontsize=labels_fontsize)
            if (i == 0) and (j == 0):
                axes[j][i].legend()
        if n_cols == 3:
            r_val, p_val, _, _ = plot_scatters(
                df=df_labeled_metrics,
                metric_names=['pca_singleview_error', 'pca_multiview_error'],
                train_frames=train_frames, split_set=split_set, distribution='OOD',
                model_types=[model_type, model_type], keypoint=keypoint, ax=axes[j][i + 1],
                add_trendline=True, markersize=5, scale='linear')
            axes[j][i + 1].set_title('r=%1.2f [p=%1.3f]' % (r_val, p_val))
            axes[j][i + 1].set_xlabel('Pose PCA (pix)', fontsize=labels_fontsize)
            axes[j][i + 1].set_ylabel('Multi-view PCA (pix)', fontsize=labels_fontsize)

    if title is not None:
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_file is not None:
        savefig(save_file)
    if display_plot:
        plt.show()
    else:
        plt.close()


def plot_metric_bars_and_scatters_video(
        df_video_metrics, models_to_compare, keypoint, train_frames,
        title=None, display_plot=True, save_file=None):
    """First row is barplots of metrics across all models, second row is scatters for two models"""

    plots = {
        'temporal_norm': 'Temporal Norm',
        'pca_singleview_error': 'Pose PCA',
    }
    if 'pca_multiview_error' in df_video_metrics.metric.unique():
        plots['pca_multiview_error'] = 'Multi-view PCA'
    plots['confidence'] = 'Confidence'

    n_videos = len(df_video_metrics.index.unique())

    # row 1: barplots of metrics for all models
    # row 2: scatterplots of metrics on OOD data for 2 models
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows, len(plots), figsize=(4 * len(plots), 3.5 * n_rows + 0.5), squeeze=False)

    for i, (metric_name, ax_title) in enumerate(plots.items()):

        metric_str = 'confidence' if metric_name == 'confidence' else 'error (pix)'

        # row 1: barplots of metrics for all models
        mask = ((df_video_metrics.metric == metric_name)
                & (df_video_metrics.train_frames == train_frames))
        sns.boxplot(
            x='model_type', y=keypoint,  # hue='model_type',
            order=['baseline', 'context', 'semi-super', 'semi-super context'],
            data=df_video_metrics[mask],
            ax=axes[0][i],
        )
        axes[0][i].set_title(ax_title)
        axes[0][i].set_ylabel(metric_str.capitalize(), fontsize=labels_fontsize)
        axes[0][i].set_xlabel('Model', fontsize=labels_fontsize)
        axes[0][i].set_xticklabels(
            ['Baseline', 'Context', 'Semi-super', 'Semi-super\nContext'], fontsize=labels_fontsize)

        # row 2: scatterplots of metrics on OOD data for 2 models
        plot_scatters(
            df=df_video_metrics, metric_names=[metric_name, metric_name],
            train_frames=train_frames, split_set=None, distribution=None,
            model_types=models_to_compare, keypoint=keypoint, ax=axes[1][i], add_diagonal=True,
            alpha=0.75)
        axes[1][i].set_title('%s (%i videos)' % (ax_title, n_videos))
        axes[1][i].set_xlabel(
            '%s model %s' % (models_to_compare[0].capitalize(), metric_str),
            fontsize=labels_fontsize)
        axes[1][i].set_ylabel(
            '%s model %s' % (models_to_compare[1].capitalize(), metric_str),
            fontsize=labels_fontsize)
        if i == 0:
            axes[1][i].legend()

    if title is not None:
        plt.subplots_adjust(top=0.95)
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()

    if save_file is not None:
        savefig(save_file)
    if display_plot:
        plt.show()
    else:
        plt.close()


def savefig(save_file):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, dpi=300)


# -------------------------------------------------------------------------------------------------
# these functions produce reports
# -------------------------------------------------------------------------------------------------
def generate_report_labeled(
    dataset_name,
    df_save_dir,
    fig_save_dir,
    train_frames_list=['75', '1'],
    file_ext='png'
):

    sns.set_style('white')

    df_labeled_metrics = pd.read_parquet(
        os.path.join(df_save_dir, "%s_labeled_metrics.pqt" % dataset_name))

    keypoint = 'mean'
    split_set = 'test'  # 'test' is only value for which InD and OOD both have results

    # plot basic metrics for each model type
    models_to_compare = ['baseline', 'semi-super context']
    for train_frames in train_frames_list:
        train_frame_str = 'full train frames' if train_frames == '1' \
            else '%s train frames' % train_frames
        title = 'Labeled data results on %s dataset (%s)' % (dataset_name, train_frame_str)
        save_file = os.path.join(
            fig_save_dir, 'labeled_metric_bars_and_scatters_keypoint=%s_trainframes=%s.%s' % (
                keypoint, train_frames, file_ext))
        plot_metric_bars_and_scatters_labeled(
            df_labeled_metrics, models_to_compare, keypoint, train_frames, split_set, pix_thresh=0,
            title=title, display_plot=False, save_file=save_file)

    # plot scatters of various metrics vs each other
    models_to_compare = ['baseline', 'context', 'semi-super', 'semi-super context']
    for train_frames in train_frames_list:
        train_frame_str = 'full train frames' if train_frames == '1' \
            else '%s train frames' % train_frames
        title = 'Labeled data results on %s dataset (%s)' % (dataset_name, train_frame_str)

        save_file = os.path.join(
            fig_save_dir, 'labeled_metric_vs_pixerror_scatters_keypoint=%s_trainframes=%s.%s' % (
                keypoint, train_frames, file_ext))
        plot_metric_vs_pixerror_scatters(
            df_labeled_metrics, models_to_compare, keypoint, train_frames, split_set,
            title=title, display_plot=False, save_file=save_file)

        save_file = os.path.join(
            fig_save_dir, 'labeled_metric_vs_metric_scatters_keypoint=%s_trainframes=%s.%s' % (
                keypoint, train_frames, file_ext))
        plot_metric_vs_metric_scatters(
            df_labeled_metrics, models_to_compare, keypoint, train_frames, split_set,
            title=title, display_plot=False, save_file=save_file)


def generate_report_video(
        dataset_name, df_save_dir, fig_save_dir, rng_seed, time_window=None, file_ext='png',
        models_to_compare=['baseline', 'semi-super context'], train_frames_list=['75', '1'],
        df_save_dir_dlc=None):

    sns.set_style('white')

    df_video_preds = pd.read_parquet(
        os.path.join(df_save_dir, "%s_video_preds.pqt" % dataset_name))
    df_video_metrics = pd.read_parquet(
        os.path.join(df_save_dir, "%s_video_metrics.pqt" % dataset_name))
    if 'dlc' in models_to_compare:
        df_video_preds = pd.concat([
            df_video_preds,
            pd.read_parquet(
                os.path.join(df_save_dir_dlc, "%s_video_preds.pqt" % dataset_name))
        ])
        df_video_metrics = pd.concat([
            df_video_metrics,
            pd.read_parquet(
                os.path.join(df_save_dir_dlc, "%s_video_metrics.pqt" % dataset_name))
        ])
    df_video_metrics_gr = df_video_metrics.groupby([
        'metric', 'video_name', 'model_path', 'rng_seed_data_pt', 'train_frames', 'model_type']
    ).mean().reset_index().set_index('video_name')

    # plot basic metrics for each model type
    keypoint = 'mean'
    for train_frames in train_frames_list:
        train_frame_str = 'full train frames' if train_frames == '1' \
            else '%s train frames' % train_frames
        title = 'Video results on %s dataset (%s)' % (dataset_name, train_frame_str)
        save_file = os.path.join(
            fig_save_dir, 'video_metric_bars_and_scatters_keypoint=%s_trainframes=%s.%s' % (
                keypoint, train_frames, file_ext))
        plot_metric_bars_and_scatters_video(
            df_video_metrics_gr, models_to_compare, keypoint, train_frames,
            title=title, display_plot=False, save_file=save_file)

    # plot traces on a single video for each keypoint
    if time_window is None:
        time_window = (0, 1000)
    for vid_name in df_video_metrics.video_name.unique():
        for keypoint in df_video_preds.columns.levels[0]:
            if keypoint in [
                "model_path", "model_type", "rng_seed_data_pt", "train_frames", "video_name"
            ]:
                continue
            else:
                for train_frames in train_frames_list:
                    save_file = os.path.join(
                        fig_save_dir, 'video_traces_keypoint=%s_trainframes=%s_vid=%s.%s' % (
                            keypoint, train_frames, vid_name, file_ext))
                    plot_traces_and_metrics(
                        df_video_metrics=df_video_metrics, df_video_preds=df_video_preds,
                        models_to_compare=models_to_compare, keypoint=keypoint, vid_name=vid_name,
                        train_frames=train_frames, rng_seed=rng_seed, time_window=time_window,
                        display_plot=False, save_file=save_file)


# -------------------------------------------------------------------------------------------------
# these are util functions for creating metric dataframes
# -------------------------------------------------------------------------------------------------
def update_col_names(df):
    old_names_0 = df.columns.levels[0]
    new_names_0 = {}
    for n in old_names_0:
        new_name = n if n.find("Unnamed") == -1 else "set"
        new_names_0[n] = new_name
    old_names_1 = df.columns.levels[1]
    new_names_1 = {}
    for n in old_names_1:
        new_name = n if n.find("Unnamed") == -1 else ""
        new_names_1[n] = new_name
    df = df.rename(columns=new_names_0, level=0)
    df = df.rename(columns=new_names_1, level=1)
    return df


def get_video_names(file_list):
    video_names = []
    for file in file_list:
        if "_pca_singleview_error" in file:
            continue
        elif "_pca_multiview_error" in file:
            continue
        elif "_temporal_norm" in file:
            continue
        else:
            video_names.append(file.replace(".csv", ""))
    return video_names


def add_model_metadata(df, model, levels):
    from diagnostics.inventory import get_model_type
    updates = {
        "model_path": model["path"],
        "rng_seed_data_pt": model["training.rng_seed_data_pt"],
        "train_frames": model["training.train_frames"],
        "model_type": get_model_type(model),
    }
    for key, val in updates.items():
        # always put the key at the top level of a multi-index
        # fill out remaining levels with empty strings
        acc_str = (key,)
        for _ in range(1, levels):
            acc_str += ("",)
        df.loc[:, acc_str] = val
