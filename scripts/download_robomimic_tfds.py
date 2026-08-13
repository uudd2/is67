#!/usr/bin/env python3
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds


tf.config.set_visible_devices([], "GPU")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/media/dm/Elements/VLANeXt_migration/data/RoboMimic_TFDS",
    )
    args = parser.parse_args()

    configs = ("lift_ph_image", "can_ph_image", "square_ph_image")
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] Preparing robomimic_ph/{config}", flush=True)
        builder = tfds.builder("robomimic_ph", config=config, data_dir=args.data_dir)
        builder.download_and_prepare()
        print(f"Completed: {builder.data_dir}", flush=True)


if __name__ == "__main__":
    main()
