# %%
# print("Importing Libraries")
import numpy as np
from pathlib import Path
import os
import torch
import matplotlib.pyplot as plt
from src.utilities.load_model import load_model
from src.utilities.dataloader import get_dataloaders
import pickle

plt.style.use("seaborn-v0_8-whitegrid")
import json


class EvaluatorZeroshot:
    def __init__(self, train_loader, test_loader, device="cpu"):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.zero_shot = True
        self.model_names = [
            "ZeroShot_C_AE_var",
            "ZeroShot_CLSP_Var",
            "CLSP_zeroshot_no_norm",
            "CLSP_AE_bootstrap_latest",
        ]
        self.variance_models = [
            "ZeroShot_C_AE_var",
            "ZeroShot_CLSP_Var",
            "CLSP_zeroshot_no_norm",
            "CLSP_AE_bootstrap_latest",
        ]
        self.setup_dictionaries
        self.set_subject_and_task(1, 1)  # Default subject and tasks
        self.set_up_model("ZeroShot_CLSP_Var")

    def setup_dictionaries(self):
        with open(f"data/processed/labels_dict.json", "r") as f:
            labels_dict = json.load(f)
        self.task_to_label = labels_dict["task_to_label"]
        self.channel_to_label = labels_dict["channel_to_label"]
        self.task_to_channels = labels_dict["task_to_channels"]
        self.label_to_task = {v: k for k, v in self.task_to_label.items()}
        self.label_to_channel = {v: k for k, v in self.channel_to_label.items()}

    def get_channels_of_interest(self, task: int):
        task_label = self.task_to_label[str(task)]
        channel_labels = self.task_to_channels[task_label]
        channels = np.array(
            [int(self.label_to_channel[label]) for label in channel_labels]
        )
        return channels

    def set_up_model(self, model_name: str):
        # check if model_name is valid
        if model_name not in self.model_names:
            raise ValueError(f"Invalid model_name: {model_name}")
        self.model, self.model_config = load_model(
            model_name, device=self.device, chkpt=None
        )
        if hasattr(self.model, "update_current_epoch"):
            self.model.update_current_epoch(200)
        self.model.eval()

        self.model_outputs_variance = (
            True if model_name in self.variance_models else False
        )

    def denorm(self, data, specific_loader=None):
        if specific_loader:
            return specific_loader.invtransform(data)
        else:
            return self.train_loader.invtransform(data)

    def norm(self, data, specific_loader=None):
        if specific_loader:
            return specific_loader.transform(data)
        else:
            return self.train_loader.transform(data)

    def get_subject_task_data_subset(self, mode="train", denorm=False):
        data_loader = self.train_loader if mode == "train" else self.test_loader
        subject_task_data_subset = data_loader.get_subject_task_specific_data(
            self.subject, self.task
        )
        if denorm:
            subject_task_data_subset = self.denorm(subject_task_data_subset)
        return subject_task_data_subset

    def denoise(self, data, normalize_input=False):
        with torch.no_grad():
            if normalize_input:
                data = self.norm(data)
            if self.model_outputs_variance:
                output, variance = self.model(data, return_var=True)
                return (
                    self.denorm(output),
                    self.denorm(variance**0.5) ** 2,
                )
            else:
                output = self.model(data)
                return self.denorm(output), None

    def get_seeded_permutations(self, data, n, seed):
        seeds = torch.randint(
            0, 2**32 - 1, (n,), generator=torch.Generator().manual_seed(seed)
        )
        permutations = torch.stack(
            [
                data[
                    torch.randperm(
                        data.shape[0], generator=torch.Generator().manual_seed(s.item())
                    )
                ]
                for s in seeds
            ],
            dim=0,
        )
        return permutations

    def metric_function(self, y_true, y_pred, metric="MSE"):
        if metric == "MSE":
            return torch.mean((y_true - y_pred) ** 2, dim=-1)
        elif metric == "R2":
            mean_true = torch.mean(y_true, dim=-1, keepdim=True)
            total_sum_of_squares = torch.sum((y_true - mean_true) ** 2, dim=-1)
            residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2, dim=-1)
            r2 = 1 - residual_sum_of_squares / total_sum_of_squares
            return r2

    def cumulative_averageing(self, mu, var, method="mean"):
        def amplitude_sorting(mu):
            # amplitude = torch.max(mu, dim=-1).values - torch.min(mu, dim=-1).values
            amplitude = torch.var(mu, dim=-1)
            sorted_indices = torch.argsort(amplitude, dim=1)
            # mu = torch.gather(mu, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, -1, mu.shape[-1]))
            return sorted_indices.unsqueeze(-1).expand(-1, -1, -1, mu.shape[-1])

        def amplitude_sorting2(var):
            # amplitude = torch.max(mu, dim=-1).values - torch.min(mu, dim=-1).values
            amplitude = var[..., 0]
            sorted_indices = torch.argsort(amplitude, dim=1)
            # mu = torch.gather(mu, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, -1, mu.shape[-1]))
            return sorted_indices.unsqueeze(-1).expand(-1, -1, -1, var.shape[-1])

        cumsum = lambda mu, w: torch.cumsum(mu * w, dim=1) / torch.cumsum(w, dim=1)
        if method == "mean" and var is None:
            return cumsum(mu, torch.ones_like(mu))
        elif method == "mean" and var is not None:
            return cumsum(mu, 1 / var)
        elif method == "trimmed_mean":
            res = (
                torch.zeros_like(mu) * torch.nan
            )  # shape (n_samples, n_trials, n_channels, n_timepoints)
            alpha = 0.1
            for N in range(1, mu.shape[1] + 1):
                sort_indices = amplitude_sorting(
                    mu[:, :N]
                )  # shape (n_samples, n_trials, n_channels, n_timepoints)
                lower_bound = int(alpha * N)
                upper_bound = N - lower_bound
                sort_indices = sort_indices[:, lower_bound:upper_bound]
                res[:, N - 1] = torch.mean(torch.gather(mu, 1, sort_indices), dim=1)
            return res
        elif method == "tanh_mean":
            res = (
                torch.zeros_like(mu) * torch.nan
            )  # shape (n_samples, n_trials, n_channels, n_timepoints)

            k = 0.1
            s = 0
            for N in range(mu.shape[1] + 1):
                # k = 10 / (N + 1)
                sort_indices = amplitude_sorting(mu[:, :N])
                mu_sorted = torch.gather(mu, 1, sort_indices)
                weights = torch.zeros_like(mu_sorted)
                # weights[:, : N // 2] = (torch.tanh(k * torch.arange(N // 2)) - s)[
                #     None, :, None, None
                # ]
                weights[:, : N // 2] += -torch.tanh(torch.tensor(k * (N // 2 - N) + s))
                weights[:, N // 2 :] = -torch.tanh(
                    k * (torch.arange(N // 2, N) - N) + s
                )[None, :, None, None]
                res[:, N - 1] = torch.sum(mu_sorted * weights, dim=1) / torch.sum(
                    weights, dim=1
                )
            res[res < 0] = 0
            return res
        elif method == "our":
            res = (
                torch.zeros_like(mu) * torch.nan
            )  # shape (n_samples, n_trials, n_channels, n_timepoints)
            k = 0.1
            s = 0
            for N in range(mu.shape[1] + 1):
                # k = 10 / (N + 1)
                sort_indices = amplitude_sorting2(var[:, :N])
                mu_sorted = torch.gather(mu, 1, sort_indices)
                weights = torch.zeros_like(mu_sorted)
                weights[:, : N // 2] += -torch.tanh(torch.tensor(k * (N // 2 - N) + s))
                weights[:, N // 2 :] = -torch.tanh(
                    k * (torch.arange(N // 2, N) - N) + s
                )[None, :, None, None]
                res[:, N - 1] = torch.sum(mu_sorted * weights, dim=1) / torch.sum(
                    weights, dim=1
                )
            res[res < 0] = 0
            return res
        # elif method == "our":
        #     res = (
        #         torch.zeros_like(mu) * torch.nan
        #     )  # shape (n_samples, n_trials, n_channels, n_timepoints)
        #     alpha = 0.1
        #     for N in range(1, mu.shape[1] + 1):
        #         sort_indices = amplitude_sorting2(
        #             var[:, :N]
        #         )  # shape (n_samples, n_trials, n_channels, n_timepoints)
        #         lower_bound = 0
        #         upper_bound = N - int(2 * alpha * N)
        #         sort_indices = sort_indices[:, lower_bound:upper_bound]
        #         res[:, N - 1] = torch.mean(torch.gather(mu, 1, sort_indices), dim=1)
        #     return res

    def calculate_evaluation_curve(
        self,
        denoise=True,
        metric="MSE",
        no_repetitions=1,
        seed=123,
        averaging_method="mean",
        return_var_curves=False,
    ):
        mu, var = self.get_mu_and_var(denoise)
        mu = self.get_seeded_permutations(mu[:, self.channels], no_repetitions, seed)
        if var is not None:
            var = self.get_seeded_permutations(
                var[:, self.channels], no_repetitions, seed
            )
        curves = self.metric_function(
            self.target[self.channels].unsqueeze(0),
            self.cumulative_averageing(mu, var, method=averaging_method),
            metric=metric,
        )
        if return_var_curves:
            var_curves = self.cumulative_averageing(var, None, method="mean")
            return (
                curves.mean(dim=0).squeeze(),
                curves.std(dim=0).squeeze(),
                var_curves.mean(dim=0).squeeze()[:, 0],
                var_curves.std(dim=0).squeeze()[:, 0],
            )
        return curves.mean(dim=0).squeeze(), curves.std(dim=0).squeeze()

    def calculate_evaluation_curve2(
        self,
        denoise=True,
        metric="MSE",
        no_repetitions=1,
        seed=123,
        averaging_method="mean",
    ):

        if not self.zero_shot:
            mu, var = self.denoise(self.train_data)
            data = self.train_data
        else:
            mu, var = self.get_mu_and_var(denoise)
            data = self.test_data
        mu = self.get_seeded_permutations(mu[:, self.channels], no_repetitions, seed)
        data = self.get_seeded_permutations(
            data[:, self.channels], no_repetitions, seed
        )
        if var is not None:
            var = self.get_seeded_permutations(
                var[:, self.channels], no_repetitions, seed
            )
        curves = self.metric_function(
            self.target[self.channels].unsqueeze(0),
            self.cumulative_averageing(data, var, method=averaging_method),
            metric=metric,
        )
        return curves.mean(dim=0).squeeze(), curves.std(dim=0).squeeze()

    def get_mu_and_var(self, denoise=True):
        if denoise:
            mu, var = (
                self.denoise(self.test_input_half_norm)
                if self.model_outputs_variance
                else (
                    self.denoise(self.test_input_half_norm),
                    torch.ones_like(self.test_input_half_norm),
                )
            )
        else:
            mu, var = self.denorm(self.test_input_half_norm), None
        return mu, var

    def set_subject_and_task(self, subject: int, task: int):
        self.subject = subject
        self.task = task
        self.channels = self.get_channels_of_interest(task)
        self.test_data_norm = self.get_subject_task_data_subset(
            mode="test", denorm=False
        )
        n = self.test_data_norm.shape[0]
        self.test_input_half_norm = self.test_data_norm[: n // 2]
        self.test_data = self.denorm(self.test_input_half_norm)
        self.input_shape = self.test_input_half_norm.shape
        self.n_test = self.input_shape[0]

        self.target = self.denorm(self.test_data_norm[n // 2 :].mean(dim=0))


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataloader",
        choices=["myCustomLoader", "myCustomLoaderZeroShot"],
        default="myCustomLoader",
    )
    parser.add_argument("--skip_tasks_up_to", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(f"reports/figures/evaluation/{args.dataloader}", exist_ok=True)
    dataloader = args.dataloader
    # %%
    # dataloader = "myCustomLoaderZeroShot"
    # set cwd to root directory
    os.chdir(Path(__file__).resolve().parents[2])
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    print("Loading Data")
    train_loader, test_loader = get_dataloaders(
        cuda=cuda, dataloader=dataloader, mode="both"
    )
    print("Successfully loaded data")
    # %%
    from tqdm import tqdm

    if dataloader == "myCustomLoader":
        evaluator = Evaluator(train_loader, test_loader, device=device)
        evaluator.set_up_model("C_AE_no_norm")
    else:
        evaluator = EvaluatorZeroshot(train_loader, test_loader, device=device)
        evaluator.set_up_model("CLSP_zero_shot_no_norm")

    for task in range(args.skip_tasks_up_to, 14):
        n_min = None
        n_max = None
        exclude_threshold = 2
        no_repetitions = 1
        metric = "R2"
        baseline_curves = []
        trimmed_mean_curves = []
        tanh_mean_curves = []
        denoised_curves = []

        for subject in tqdm(sorted(test_loader.unique_subjects)[n_min:n_max]):
            evaluator.set_subject_and_task(subject, task)
            if evaluator.n_test < exclude_threshold:
                continue
            mu, std = evaluator.calculate_evaluation_curve(
                metric=metric, denoise=False, no_repetitions=no_repetitions
            )
            baseline_curves.append(mu)
            mu, std = evaluator.calculate_evaluation_curve(
                metric=metric,
                denoise=False,
                averaging_method="trimmed_mean",
                no_repetitions=no_repetitions,
            )
            trimmed_mean_curves.append(mu)
            mu, std = evaluator.calculate_evaluation_curve(
                metric=metric,
                denoise=False,
                averaging_method="tanh_mean",
                no_repetitions=no_repetitions,
            )
            tanh_mean_curves.append(mu)
            mu, std = evaluator.calculate_evaluation_curve(
                metric=metric, denoise=True, no_repetitions=no_repetitions
            )
            denoised_curves.append(mu)

        def pad_arrays(arrays):
            max_length = max(len(arr) for arr in arrays)
            padded_arrays = [
                np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan)
                for arr in arrays
            ]
            matrix = np.array(padded_arrays)
            return matrix

        baseline_curves = pad_arrays(baseline_curves)
        trimmed_mean_curves = pad_arrays(trimmed_mean_curves)
        tanh_mean_curves = pad_arrays(tanh_mean_curves)
        denoised_curves = pad_arrays(denoised_curves)

        # remove columns that contain at least one nan
        mask = np.isnan(baseline_curves).any(axis=0)
        baseline_curves = baseline_curves[:, ~mask]
        trimmed_mean_curves = trimmed_mean_curves[:, ~mask]
        tanh_mean_curves = tanh_mean_curves[:, ~mask]
        mask = np.isnan(denoised_curves).any(axis=0)
        denoised_curves = denoised_curves[:, ~mask]

        row_means = np.nanmean(baseline_curves, axis=1)
        sort_indices = np.argsort(row_means)
        baseline_curves_sorted = baseline_curves[sort_indices]
        denoised_curves_sorted = denoised_curves[sort_indices]
        trimmed_mean_curves_sorted = trimmed_mean_curves[sort_indices]

        # dump curves to pickle
        #
        os.makedirs(f"data/processed/evaluation_curves/{dataloader}", exist_ok=True)
        with open(
            f"data/processed/evaluation_curves/{dataloader}/task_{task}_baseline.pkl",
            "wb",
        ) as f:
            pickle.dump(baseline_curves, f)
        with open(
            f"data/processed/evaluation_curves/{dataloader}/task_{task}_denoised.pkl",
            "wb",
        ) as f:
            pickle.dump(denoised_curves, f)
        with open(
            f"data/processed/evaluation_curves/{dataloader}/task_{task}_trimmed_mean.pkl",
            "wb",
        ) as f:
            pickle.dump(trimmed_mean_curves, f)
        with open(
            f"data/processed/evaluation_curves/{dataloader}/task_{task}_tanh_mean.pkl",
            "wb",
        ) as f:
            pickle.dump(tanh_mean_curves, f)

        #
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        cmap = plt.get_cmap("brg")
        colors = cmap(np.linspace(0, 1, len(baseline_curves)))
        plt.gca().set_prop_cycle(color=colors)
        mu = np.nanmean(baseline_curves, axis=0)
        ax.plot(mu, label="Baseline", color="r", linestyle="-", linewidth=3)
        mu = np.nanmean(denoised_curves, axis=0)
        ax.plot(mu, label="Denoised", color="g", linestyle="-", linewidth=3)
        mu = np.nanmean(trimmed_mean_curves, axis=0)
        ax.plot(mu, label="Trimmed Mean", color="b", linestyle="-", linewidth=3)
        mu = np.nanmean(tanh_mean_curves, axis=0)
        ax.plot(mu, label="Tanh Mean", color="y", linestyle="-", linewidth=3)
        ax.legend()
        ax.set_ylim(-4, 1)
        ax.set_xlim(0, len(np.nanmean(baseline_curves, axis=0) != np.nan))
        plt.tight_layout()
        plt.savefig(f"reports/figures/evaluation/{dataloader}/task_{task}_{metric}.png")
        plt.close()


# %%
