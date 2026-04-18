from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import shap


ICE_FEATURE_ORDER = [
    "carbonization_temperature_C",
    "ssa_m2_g",
    "current_density_mA_g",
    "d002_nm",
    "id_ig",
    "electrolyte_type",
]

PLATEAU_FEATURE_ORDER = [
    "carbonization_temperature_C",
    "ssa_m2_g",
    "d002_nm",
    "current_density_mA_g",
    "id_ig",
    "electrolyte_type",
]

FEATURE_LABELS = {
    "carbonization_temperature_C": "Carbonization\ntemperature",
    "ssa_m2_g": "SSA",
    "current_density_mA_g": "Current density",
    "d002_nm": r"d$_{002}$",
    "id_ig": r"I$_D$/I$_G$",
    "electrolyte_type": "Electrolyte",
}


def load_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    y_true_col = cols.get("y_true")
    y_pred_col = cols.get("y_pred")
    if y_true_col is None or y_pred_col is None:
        raise ValueError(f"{path} must contain y_true and y_pred columns.")
    return df.rename(columns={y_true_col: "y_true", y_pred_col: "y_pred"})[["y_true", "y_pred"]]


def add_diag_lines(ax, x_min: float, x_max: float, mae: float, color: str = "#c77728") -> None:
    xs = [x_min, x_max]
    ax.plot(xs, xs, color=color, linewidth=1.2)
    ax.plot(xs, [x_min + mae, x_max + mae], color=color, linewidth=1.0, linestyle="--", alpha=0.9)
    ax.plot(xs, [x_min - mae, x_max - mae], color=color, linewidth=1.0, linestyle="--", alpha=0.9)


def plot_panel_s2(
    ax,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    train_color: str,
    test_color: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    r2_train = r2_score(train_df["y_true"], train_df["y_pred"])
    mae_train = mean_absolute_error(train_df["y_true"], train_df["y_pred"])

    ax.scatter(test_df["y_true"], test_df["y_pred"], s=13, color=test_color, edgecolor="none", alpha=0.85, label="Test")
    ax.scatter(train_df["y_true"], train_df["y_pred"], s=13, color=train_color, edgecolor="none", alpha=0.9, label="Train")

    if xlim is None:
        x_min = min(train_df["y_true"].min(), test_df["y_true"].min(), train_df["y_pred"].min(), test_df["y_pred"].min())
        x_max = max(train_df["y_true"].max(), test_df["y_true"].max(), train_df["y_pred"].max(), test_df["y_pred"].max())
        pad = 0.05 * (x_max - x_min)
        xlim = (x_min - pad, x_max + pad)
    if ylim is None:
        ylim = xlim

    add_diag_lines(ax, xlim[0], xlim[1], mae_train)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.3, borderpad=0.2)
    ax.text(
        0.98,
        0.06,
        rf"R$^2$={r2_train:.3f}" + f"\nMAE={mae_train:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)



def save_figure_s2(reviewer_dir: Path, out_path: Path) -> None:
    train_ice = load_pred_csv(reviewer_dir / "rf_predictions_train_cv_ice.csv")
    test_ice = load_pred_csv(reviewer_dir / "rf_predictions_test_ice.csv")
    train_plateau = load_pred_csv(reviewer_dir / "rf_predictions_train_cv_plateau.csv")
    test_plateau = load_pred_csv(reviewer_dir / "rf_predictions_test_plateau.csv")

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.4), constrained_layout=True)

    plot_panel_s2(
        axes[0],
        train_ice,
        test_ice,
        xlabel="ICE_Real (%)",
        ylabel="ICE_Predict (%)",
        train_color="#4e79a7",
        test_color="#f08a84",
        xlim=(-2.0, 3.0),
        ylim=(-2.0, 3.0),
    )
    axes[0].text(-0.18, 1.02, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold")

    plot_panel_s2(
        axes[1],
        train_plateau,
        test_plateau,
        xlabel=r"Plateau capacity_Real (mAh g$^{-1}$)",
        ylabel=r"Plateau capacity_Predict (mAh g$^{-1}$)",
        train_color="#0b7ea1",
        test_color="#d9c2a3",
        xlim=(0.0, 300.0),
        ylim=(0.0, 300.0),
    )
    axes[1].text(-0.18, 1.02, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def render_shap_panel(csv_path: Path, feature_order: list[str], cmap, temp_png: Path) -> None:
    df = pd.read_csv(csv_path)
    shap_cols = [f"{f}_shap" for f in feature_order]

    shap_values = df[shap_cols].to_numpy()
    X = df[feature_order].copy()
    X.columns = [FEATURE_LABELS[f] for f in feature_order]

    plt.figure(figsize=(7.0, 3.6))
    shap.summary_plot(
        shap_values=shap_values,
        features=X,
        feature_names=list(X.columns),
        sort=False,
        show=False,
        cmap=cmap,
        max_display=len(feature_order),
        color_bar=True,
    )
    plt.gcf().savefig(temp_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(plt.gcf())



def save_figure_s3(shap_dir: Path, out_path: Path) -> None:
    ice_csv = shap_dir / "ice_test_used_for_shap.csv"
    plateau_csv = shap_dir / "plateau_test_used_for_shap.csv"

    plateau_cmap = LinearSegmentedColormap.from_list(
        "plateau_cmap",
        ["#b7e4c7", "#7fdac1", "#2c7fb8", "#1f1b5b"],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ice_png = tmpdir / "ice_panel.png"
        plateau_png = tmpdir / "plateau_panel.png"

        render_shap_panel(ice_csv, ICE_FEATURE_ORDER, plt.get_cmap("coolwarm_r"), ice_png)
        render_shap_panel(plateau_csv, PLATEAU_FEATURE_ORDER, plateau_cmap, plateau_png)

        ice_img = mpimg.imread(ice_png)
        plateau_img = mpimg.imread(plateau_png)

        fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.6), constrained_layout=True)
        for ax, img, label in [(axes[0], ice_img, "a"), (axes[1], plateau_img, "b")]:
            ax.imshow(img)
            ax.axis("off")
            ax.text(0.01, 0.97, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Supplementary Fig. 2 and Fig. 3 from saved best-RF outputs.")
    parser.add_argument("--reviewer-dir", type=Path, default=Path("results/reviewer"))
    parser.add_argument("--shap-dir", type=Path, default=Path("outputs/shap_data"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_figure_s2(args.reviewer_dir, args.out_dir / "fig_S2_rf_best.png")
    save_figure_s3(args.shap_dir, args.out_dir / "fig_S3_shap_rf_best.png")

    print(f"Saved: {args.out_dir / 'fig_S2_rf_best.png'}")
    print(f"Saved: {args.out_dir / 'fig_S3_shap_rf_best.png'}")


if __name__ == "__main__":
    main()
