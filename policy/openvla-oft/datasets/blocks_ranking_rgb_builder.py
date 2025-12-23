from typing import Iterator, Tuple, Any
import os
import h5py
import glob
import numpy as np
import tensorflow_datasets as tfds
import random
from datasets.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    print(f"[INFO] Generating examples from {len(paths)} paths")
    for path in paths:
        print(f"[INFO] Parsing file: {path}")
        with h5py.File(path, "r") as f:
            required_keys = [
                "/relative_action",
                "/head_camera_image",
                "/left_wrist_image",
                "/right_wrist_image",
                "/low_cam_image",
                "/action",
                "/seen",
            ]
            if not all(k in f for k in required_keys):
                for key in required_keys:
                    if key not in f:
                        print(f"[ERROR] Missing key: {key} in {path}")
                print(f"[WARNING] Missing expected keys in {path}, skipping")
                continue
            T = f["/action"].shape[0]
            actions = f["/action"][1:].astype(np.float32)  # (T-1, 14)
            head = f["/head_camera_image"][ : T-1 ].astype(np.uint8)
            left = f["/left_wrist_image"][ : T-1].astype(np.uint8)
            right = f["/right_wrist_image"][ :T-1].astype(np.uint8)
            low = f["/low_cam_image"][ : T-1].astype(np.uint8)
            states = f["/action"][: T - 1].astype(np.float32)  # (T-1, 14)
            seen = [
                s.decode("utf-8") if isinstance(s, bytes) else s for s in f["/seen"][()]
            ]
            T -= 1

            if not seen:
                print(f"[ERROR] No 'seen' instructions found in {path}")
                continue

            if not (
                head.shape[0]
                == left.shape[0]
                == right.shape[0]
                == low.shape[0]
                == T
                == states.shape[0]
            ):
                print(f"[ERROR] Data length mismatch in {path}")
                continue

            instruction = seen

            steps = []
            for i in range(T):
                step = {
                    "observation": {
                        "image": head[i],
                        "left_wrist_image": left[i],
                        "right_wrist_image": right[i],
                        "low_cam_image": low[i],
                        "state": states[i],
                    },
                    "action": actions[i],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == T - 1 else 0.0),
                    "is_first": np.bool_(i == 0),
                    "is_last": np.bool_(i == T - 1),
                    "is_terminal": np.bool_(i == T - 1),
                    "language_instruction": instruction,
                }
                steps.append(step)

            print(f"[INFO] Yielding {len(steps)} steps from {path}")
            yield path, {"steps": steps, "episode_metadata": {"file_path": path}}


class aloha_blocks_ranking_rgb(MultiThreadedDatasetBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release for RoboTwin blocks_ranking_rgb dataset.",
    }

    N_WORKERS = 1
    MAX_PATHS_IN_MEMORY = 100
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Head camera RGB observation.",
                                    ),
                                    "left_wrist_image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Left wrist camera RGB observation.",
                                    ),
                                    "right_wrist_image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Right wrist camera RGB observation.",
                                    ),
                                    "low_cam_image": tfds.features.Image(
                                        shape=(480, 640, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Low camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(14,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [7x robot joint angles, 7x robot joint velocities].",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(14,),
                                dtype=np.float32,
                                doc="Robot action, consists of [7x joint velocities, 7x joint torques].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {"file_path": tfds.features.Text(doc="Path to the original data file.")}
                    ),
                }
            )
        )

    def _split_paths(self):
        # 指定数据路径
        data_dir = "/mnt/nvme1/shihuiz/robotwin/blocks_ranking_rgb/demo_clean/data"
        all_paths = glob.glob(os.path.join(data_dir, "*.hdf5"))
        
        print(f"[INFO] Found {len(all_paths)} HDF5 files in {data_dir}")
        
        if len(all_paths) == 0:
            raise ValueError(f"No HDF5 files found in {data_dir}")
        
        # 按照 train/val 分割（这里简单地 96% train, 4% val）
        random.shuffle(all_paths)
        split_idx = int(0.96 * len(all_paths))
        
        return {
            "train": all_paths[:split_idx],
            "val": all_paths[split_idx:],
        }
