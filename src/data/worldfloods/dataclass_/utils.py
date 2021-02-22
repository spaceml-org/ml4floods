from dataclasses import dataclass, asdict
import json
import pickle


def load_dataclass_json(dc: dataclass, file_path: str) -> dataclass:
    with open(file_path, "r") as f:
        t_dict = json.load(f)

        return dc(**t_dict)


def save_dataclass_json(dc: dataclass, file_path: str) -> None:
    with open(file_path, "w+") as f:
        json.dump(asdict(dc), f)

        return None


def save_dataclass_pickle(dc: dataclass, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(dc, f)

        return None


def load_dataclass_pickle(file_path: str) -> dataclass:
    with open(file_path, "rb") as f:
        return pickle.load(f)