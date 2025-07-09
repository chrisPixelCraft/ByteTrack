from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger
from yolox.evaluators import MOTEvaluator

import argparse
import os
import random
import warnings
import glob
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


# evaluate MOTA for NTU-MTMC dataset
results_folder = 'YOLOX_outputs/yolox_m_mix_det/track_results'
mm.lap.default_solver = 'lap'

# For NTU-MTMC, we don't use the _val_half suffix
gt_type = ''
print('gt_type (empty for NTU-MTMC)', gt_type)

# Find ground truth files for NTU-MTMC cameras
gtfiles = glob.glob(
    os.path.join('datasets/mot/train', 'Cam*/gt/gt{}.txt'.format(gt_type)))
print('gt_files:', gtfiles)

# Find tracking result files
tsfiles = [f for f in glob.glob(os.path.join(results_folder, 'Cam*.txt')) if not os.path.basename(f).startswith('eval')]
print('tracking_files:', tsfiles)

logger.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
logger.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
logger.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
logger.info('Loading files.')

# Load ground truth files (extract camera name from path)
gt = OrderedDict()
for f in gtfiles:
    # Extract camera name (e.g., 'Cam1' from 'datasets/mot/train/Cam1/gt/gt.txt')
    cam_name = Path(f).parts[-3]
    gt[cam_name] = mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)

# Load tracking results (extract camera name from filename)
ts = OrderedDict()
for f in tsfiles:
    # Extract camera name (e.g., 'Cam1' from 'Cam1.txt')
    cam_name = os.path.splitext(Path(f).parts[-1])[0]
    ts[cam_name] = mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=-1.0)

print(f"Ground truth cameras: {list(gt.keys())}")
print(f"Tracking result cameras: {list(ts.keys())}")

mh = mm.metrics.create()
accs, names = compare_dataframes(gt, ts)

logger.info('Running metrics')
metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked',
            'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses',
            'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)

# Normalize certain metrics
div_dict = {
    'num_objects': ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations'],
    'num_unique_objects': ['mostly_tracked', 'partially_tracked', 'mostly_lost']}
for divisor in div_dict:
    for divided in div_dict[divisor]:
        summary[divided] = (summary[divided] / summary[divisor])

fmt = mh.formatters
change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mostly_tracked',
                    'partially_tracked', 'mostly_lost']
for k in change_fmt_list:
    fmt[k] = fmt['mota']
print(mm.io.render_summary(summary, formatters=fmt, namemap=mm.io.motchallenge_metric_names))

# Also compute standard MOT metrics
metrics = mm.metrics.motchallenge_metrics + ['num_objects']
summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
logger.info('Completed')