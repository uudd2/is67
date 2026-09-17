"""Cache each unique LIBERO frame once, with suite-qualified identifiers."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/home/dm/QWENLA/VLANeXt_migration/VLANeXt')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from src.datasets.libero_act import LiberoAct
from src.models.edar_lite import EDARFeatureCache, FrozenDINOFeatureExtractor

CACHE = Path('/home/dm/QWENLA/VLANeXt_migration/data/edar_cache/libero4_dinov2large_256_grid8')
SUITES = ('libero_spatial_no_noops', 'libero_object_no_noops', 'libero_goal_no_noops', 'libero_10_no_noops')


def main():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    assert shutil.disk_usage(ROOT).free > 80 * 1024**3, 'Need at least 80 GiB free before caching'
    device = torch.device('cuda')
    extractor = FrozenDINOFeatureExtractor(str(ROOT / 'pretrained/dinov2-large')).to(device)
    cache = EDARFeatureCache(CACHE, extractor.metadata(), create=True)
    print(json.dumps({'event': 'cache_start', 'cache': str(CACHE), 'suites': SUITES,
                      'dtype': 'float16', 'feature_shape': [64, 1024],
                      'compute_dtype': 'float32', 'gpu': torch.cuda.get_device_name(0)}), flush=True)
    totals = {}
    for suite in SUITES:
        started = time.time()
        seen = set()
        written = 0
        dataset = LiberoAct(
            data_path=f'/media/dm/Elements/VLANeXt_migration/data/LIBERO_modified/{suite}/1.0.0',
            dataset_name=suite, history_len=1, future_len=8, full_sequence=True,
            input_modality='image', view_mode='single', load_future_image=True,
            future_image_mode='horizon', strict_future_horizon=True, buffer_size=1,
        )
        print(json.dumps({'event': 'suite_start', 'suite': suite}), flush=True)
        for index, batch in enumerate(DataLoader(dataset, batch_size=32, num_workers=0)):
            ids = list(batch['frame_id']) + list(batch['future_frame_id'])
            images = torch.cat([batch['image'], batch['future_image']], dim=0)
            missing_ids, missing_indices = [], []
            for row, frame in enumerate(ids):
                qualified = f'{suite}/{frame}'
                if qualified in seen:
                    continue
                seen.add(qualified)
                if not cache.path_for(qualified).exists():
                    missing_ids.append(qualified)
                    missing_indices.append(row)
            for offset in range(0, len(missing_ids), 32):
                keys = missing_ids[offset:offset+32]
                selected = images[missing_indices[offset:offset+32]].to(device)
                features = extractor(selected).to(torch.float16).cpu()
                assert features.shape[1:] == (64, 1024)
                assert torch.isfinite(features).all(), 'Nonfinite cached features'
                for key, feature in zip(keys, features):
                    cache.put(key, feature)
                    written += 1
            if index % 32 == 0:
                print(json.dumps({'event': 'progress', 'suite': suite, 'batches': index+1,
                                  'unique_frames': len(seen), 'new_files': written,
                                  'elapsed_s': round(time.time()-started, 1)}), flush=True)
        assert seen, f'No valid frames from {suite}'
        totals[suite] = {'unique_frames': len(seen), 'new_files': written,
                         'elapsed_s': round(time.time()-started, 1)}
        print(json.dumps({'event': 'suite_done', 'suite': suite, **totals[suite]}), flush=True)
        temporary = CACHE / 'progress.json.tmp'
        temporary.write_text(json.dumps(totals, indent=2))
        temporary.replace(CACHE / 'progress.json')
    (CACHE / 'COMPLETE.json').write_text(json.dumps(totals, indent=2))
    print(json.dumps({'event': 'cache_complete', 'totals': totals}), flush=True)


if __name__ == '__main__':
    main()
