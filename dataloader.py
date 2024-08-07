from collections import defaultdict
import numpy as np
from torch.utils.data import IterableDataset
import torch
from torch.utils.data import DataLoader
from collections import defaultdict


def get_dataloaders(cuda=True, mode="both"):
    with torch.no_grad():
        data_dict = torch.load("data/processed/simple_data.pt")
        normalizer = lambda x: x * 4e4
        data_dict["data"] = normalizer(data_dict["data"])

        targets = torch.load("data_preparation/processed/targets_bootstrap_200.pt")
        for key in targets:
            targets[key] = normalizer(targets[key])

        dataloader_module = myCustomLoaderZeroShot
        if mode == "both":

            loader = dataloader_module(
                data_dict,
                targets=targets,
                split="train",
                bootstrap=True,
                cuda=cuda,
            )
            test_loader = dataloader_module(
                data_dict,
                targets=targets,
                split="test",
                cuda=cuda,
                bootstrap=True,
            )
            return loader, test_loader
        else:
            return dataloader_module(
                data_dict,
                targets=targets,
                split=mode,
                bootstrap=True,
                cuda=cuda,
            )


class DelegatedLoader(IterableDataset):
    def __init__(self, loader, property=None, batch_size=None, length=None):
        self.loader = loader
        self._property = property
        self._batch_size = batch_size
        self._length = length

    def __len__(self):
        if self._batch_size is not None or self._property is not None:
            if self._length is not None:
                if self._batch_size is not None:
                    return self._length // self._batch_size
                return self._length
            return None
        return self.size

    def __iter__(self):
        if self._batch_size is not None or self._property is not None:
            if self._batch_size is not None:
                return self.loader.batch_iterator(self._batch_size, self._length)
            elif self._property is not None:
                return self.loader.property_iterator(self._property, self._length)
        else:
            return self.loader.iterator()


class CustomLoader:
    """from CSLP-AE"""

    def __init__(
        self, data_dict, split="train", cuda=True, bootstrap=False
    ):  # my addition
        self.bootstrap = bootstrap  # my addition
        self.split = split
        self.data = data_dict["data"]
        self.size = len(self.data)
        self.subjects = data_dict["subjects"].numpy()
        self.tasks = data_dict["tasks"].numpy()
        self.runs = (
            data_dict["runs"].clamp(min=1).numpy()
        )  # temporary fix for ERN+LRP run labels
        if cuda:
            self.data_mean = data_dict["data_mean"].detach().clone().contiguous().cuda()
            self.data_std = data_dict["data_std"].detach().clone().contiguous().cuda()
        else:
            self.data_mean = data_dict["data_mean"].detach().clone().contiguous()
            self.data_std = data_dict["data_std"].detach().clone().contiguous()

        dev_splits = [4, 7, 27, 33]
        test_splits = [5, 14, 15, 20, 22, 23, 26, 29]
        train_splits = [
            1,
            2,
            3,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
            16,
            17,
            18,
            19,
            21,
            24,
            25,
            28,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
        ]

        if split == "dev":
            self.unique_subjects = dev_splits
        elif split == "test":
            self.unique_subjects = test_splits
        elif split == "train":
            self.unique_subjects = train_splits
        elif split == "N170":
            self.unique_subjects = list(range(1, 41))
        else:
            raise ValueError("Invalid split")

        self.task_to_label = data_dict["labels"]
        self.run_labels = ["ERN+LRP", "MMN", "N2pc", "N170", "N400", "P3"]
        self.unique_tasks = list(self.task_to_label.keys())
        self.unique_runs = list(range(len(self.run_labels)))
        if split != "N170":
            self.unique_tasks = [
                t for t, l in self.task_to_label.items() if not l.startswith("N170")
            ]
            self.unique_runs = [
                r + 1 for r, l in enumerate(self.run_labels) if l != "N170"
            ]
            self.paradigms = [[0, 1], [2, 3], [4, 5], [6, 7], [10, 11], [12, 13]]
        else:
            self.unique_tasks = [
                t for t, l in self.task_to_label.items() if l.startswith("N170")
            ]
            self.unique_runs = [
                r + 1 for r, l in enumerate(self.run_labels) if l == "N170"
            ]
            self.paradigms = [[8, 9]]

        self.subject_indices = {s: [] for s in self.unique_subjects}
        self.task_indices = {t: [] for t in self.unique_tasks}
        self.run_indices = {r: [] for r in self.unique_runs}

        self.total_samples = 0
        data_indices = []
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            if s not in self.subject_indices:
                continue
            if t not in self.task_indices:
                continue
            if r not in self.run_indices:
                continue
            data_indices.append(i)
        self.data = self.data[data_indices].float().contiguous().detach().clone()
        self.data_indices = data_indices
        self.subjects = np.ascontiguousarray(self.subjects[data_indices])
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.runs = np.ascontiguousarray(self.runs[data_indices])
        self.size = len(self.data)

        self.full_indices = defaultdict(lambda: defaultdict(list))
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            self.subject_indices[s].append(i)
            self.task_indices[t].append(i)
            self.run_indices[r].append(i)
            self.full_indices[s][t].append(i)

    def reset_sample_counts(self):
        self.total_samples = 0

    def get_dataloader(
        self, num_total_samples=None, batch_size=None, property=None, random_sample=True
    ):
        delegated_loader = DelegatedLoader(
            self,
            property=property,
            batch_size=batch_size if random_sample else None,
            length=num_total_samples,
        )
        if not random_sample and batch_size is not None:
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=True)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=True)

    def sample_by_condition(self, subjects, tasks):
        samples = []
        for s, t in zip(subjects, tasks):
            i = np.random.choice(self.full_indices[s][t])
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
        return self.data[samples]

    def sample_by_property(self, property):
        property = property.lower()
        if property.startswith("s"):
            property_indices = self.subject_indices
        elif property.startswith("t"):
            property_indices = self.task_indices
        elif property.startswith("r"):
            property_indices = self.run_indices
        else:
            raise ValueError("Invalid property")

        samples = []
        for indices in property_indices.values():
            i = np.random.choice(indices)
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
        return (
            samples,
            self.data[samples],
            self.subjects[samples],
            self.tasks[samples],
            self.runs[samples],
        )

    def sample_batch(self, batch_size):
        samples = np.random.randint(0, self.size, size=batch_size)
        self.total_samples += batch_size
        return (
            samples,
            self.data[samples],
            self.subjects[samples],
            self.tasks[samples],
            self.runs[samples],
        )

    def iterator(self):
        for i in range(self.size):
            self.total_samples += 1
            yield i, self.data[i], self.subjects[i], self.tasks[i], self.runs[i]

    def batch_iterator(self, batch_size, length):
        num_samples = 0
        while True:
            if length is not None and num_samples + batch_size >= length:
                break
            yield self.sample_batch(batch_size)
            num_samples += batch_size

    def property_iterator(self, property, length):
        num_samples = 0
        num_per = 0
        while True:
            if length is not None and num_samples + num_per >= length:
                break
            yield self.sample_by_property(property)
            if length is not None:
                if num_per == 0:
                    property = property.lower()
                    if property.startswith("s"):
                        num_per = len(self.subject_indices)
                    elif property.startswith("t"):
                        num_per = len(self.task_indices)
                    elif property.startswith("r"):
                        num_per = len(self.run_indices)
                    else:
                        raise ValueError("Invalid property")
                num_samples += num_per


class myCustomLoaderZeroShot(CustomLoader):
    """used in stEEG2ERP"""

    def __init__(
        self,
        data_dict,
        targets=None,
        split="train",
        cuda=True,
        bootstrap=False,
    ):
        super().__init__(data_dict, split, cuda, bootstrap)
        self.bootstrap = bootstrap
        self.targets = targets
        for key in self.targets:
            self.targets[key] = self.targets[key].float().contiguous().detach().clone()

    def transform(self, x):
        return x * 4e4

    def invtransform(self, x):
        return x / 4e4

    def sample_batch(self, batch_size):
        samples = np.random.randint(0, self.size, size=batch_size)
        self.total_samples += batch_size

        subjects = self.subjects[samples]
        tasks = self.tasks[samples]

        if self.bootstrap:
            targets = self.get_batch_bootstrap_target(subjects, tasks, batch_size)
        else:
            targets = self.get_batch_target(subjects, tasks, batch_size)

        return (
            samples,
            self.data[samples],
            subjects,
            tasks,
            self.runs[samples],
            targets,
        )

    def get_bootstrap_target(self, subject, task):
        # get random number between 0 and 999
        idx = np.random.randint(0, 199)
        return self.targets[f"{subject},{task}"][idx, ...]

    def get_batch_bootstrap_target(self, subjects, tasks, batch_size):
        # get random numbers between 0 and 999
        idxs = np.random.randint(0, 200, size=batch_size)
        targets = torch.zeros(batch_size, self.data.shape[1], self.data.shape[2])
        for i, (s, t, idx) in enumerate(zip(subjects, tasks, idxs)):
            targets[i, ...] = self.targets[f"{s.item()},{t.item()}"][idx, ...]
        return targets

    def get_target(self, subject, task):
        return self.targets[f"{subject},{task}"]

    def get_batch_target(self, subjects, tasks, batch_size):
        targets = torch.zeros(batch_size, self.data.shape[1], self.data.shape[2])
        for i, (s, t) in enumerate(zip(subjects, tasks)):
            targets[i, ...] = self.targets[f"{s.item()},{t.item()}"]
        return targets

    def iterator(self):
        for i in range(self.size):
            self.total_samples += 1
            subject = self.subjects[i]
            task = self.tasks[i]
            if self.bootstrap:
                target = self.get_bootstrap_target(subject, task)
            else:
                target = self.get_target(subject, task)

            yield i, self.data[i], subject, task, self.runs[i], target

    def batch_iterator(self, batch_size, length):
        num_samples = 0
        while True:
            if length is not None and num_samples + batch_size >= length:
                break
            yield self.sample_batch(batch_size)
            num_samples += batch_size

    def get_subject_task_specific_data(self, subject: int, task: int):
        return self.data[(self.subjects == subject) * (self.tasks == task), ...]

    def sample_by_property(self, property):
        property = property.lower()
        if property.startswith("s"):
            property_indices = self.subject_indices
        elif property.startswith("t"):
            property_indices = self.task_indices
        elif property.startswith("r"):
            property_indices = self.run_indices
        else:
            raise ValueError("Invalid property")

        samples = []
        for indices in property_indices.values():
            i = np.random.choice(indices)
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)

        s, t = self.subjects[samples], self.tasks[samples]
        if self.bootstrap:
            targets = self.get_batch_bootstrap_target(s, t, len(samples))
        else:
            targets = self.get_batch_target(s, t, len(samples))

        return (samples, self.data[samples], s, t, self.runs[samples], targets)
