#!/usr/bin/env python3
"""Run native VITA evaluation with AV-ALOHA opened as a disk-backed Zarr store."""

import runpy

from gym_av_aloha.common.replay_buffer import ReplayBuffer


def _open_replay_buffer_from_disk(cls, zarr_path, **kwargs):
    del kwargs
    return cls.create_from_path(zarr_path, mode="r")


ReplayBuffer.copy_from_path = classmethod(_open_replay_buffer_from_disk)
runpy.run_module("flare.eval", run_name="__main__")
