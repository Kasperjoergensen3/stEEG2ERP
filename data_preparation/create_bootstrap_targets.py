import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# add seed
np.random.seed(42)
torch.manual_seed(42)

NO_SAMPLES_PER_SUBJECT_TASK = 200

# load simple data
data_dict = torch.load("data/processed/simple_data.pt")

DATA = data_dict["data"]
SUBJECTS = data_dict["subjects"]
TASKS = data_dict["tasks"]
UNIQUE_SUBJECTS = torch.unique(SUBJECTS)
UNIQUE_TASKS = torch.unique(TASKS)


def get_subject_task_specific_data(subject: int, task: int):
    return DATA[(SUBJECTS == subject) * (TASKS == task), ...]


def generate_bootstrap_sample(subject: int, task: int):
    data = get_subject_task_specific_data(subject, task)
    n_samples = data.shape[0]
    bootstrap_indices = np.random.choice(data.shape[0], n_samples, replace=True)
    return data[bootstrap_indices, ...].mean(0)


target_dict = {}
for i in tqdm(UNIQUE_SUBJECTS):
    for j in tqdm(UNIQUE_TASKS):
        samples = [
            generate_bootstrap_sample(i, j) for _ in range(NO_SAMPLES_PER_SUBJECT_TASK)
        ]
        # stack samples along new dimension
        target_dict[f"{i},{j}"] = torch.stack(samples, dim=0)

torch.save(
    target_dict, f"data/processed/targets_bootstrap_{NO_SAMPLES_PER_SUBJECT_TASK}.pt"
)
