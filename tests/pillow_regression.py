"""Compare actual FREA image helpers without importing CARLA/torch.

Run from any directory: python tests/pillow_regression.py --record /tmp/baseline.json
Then, in the candidate environment: python tests/pillow_regression.py --compare /tmp/baseline.json
AST extraction executes the original function/class bodies; it does not test simulator
imports, end-to-end behavior, or whether security vulnerabilities are fixed.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def helpers(path, names):
    source = ast.parse((ROOT / path).read_text())
    nodes = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    namespace = dict(np=np, cv2=cv2, Image=Image, ImageDraw=ImageDraw,
                     ImageOps=ImageOps, math=math, Path=Path, time=time)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / path), 'exec'), namespace)
    return namespace


def fingerprint(array):
    array = np.ascontiguousarray(array)
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def snapshots():
    result = {}
    bev = helpers('frea/agent/agent_utils/viz_tokens_bev.py', ['create_BEV', 'get_coords_BB'])
    vehicles = [[0, 0., 0., 0., 18., 4., 2.], [0, 8., -3., 37., 7.2, 4.5, 2.]]
    route = [[0, 12., 1., -15., 0., 6., 2.]]
    for extent in (True, False):
        for speed in (True, False):
            for remove in (True, False):
                image = bev['create_BEV'](vehicles, route, extent, False,
                                         use_speed_for_whole_vehicle=speed, remove_speed=remove)
                assert image.shape == (300, 300, 3) and np.isfinite(image).all()
                if remove:
                    assert not image[:, :, 1].any()
                result['bev_%s_%s_%s' % (extent, speed, remove)] = fingerprint(image)

    mask = helpers('frea/agent/agent_utils/explainability_utils.py', ['get_masked_viz_3rd_person'])
    bgr = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    for label in (0, 10, 7):
        output = mask['get_masked_viz_3rd_person'](bgr, np.full((12, 16), label))
        assert output.shape == bgr.shape and output.dtype == np.uint8
        if label in (0, 10):
            np.testing.assert_array_equal(output, bgr[:, :, ::-1])
        result['mask_%d' % label] = fingerprint(output)

    plotter = helpers('frea/agent/expert/nav_planner.py', ['Plotter'])['Plotter'](64)
    plotter.dot(np.array([1., 2.]), np.zeros(2), color=(12, 34, 56))
    result['route_plot'] = fingerprint(np.asarray(plotter.img))

    agent = helpers('frea/agent/PlanT/PlanT_agent.py', ['create_BEV', 'get_coords', 'get_coords_BB'])
    labels = [dict(position=[0., 0.], yaw=0., extent=[1., 2., 4.], **{'class': 'Car', 'speed': 2.}),
              dict(position=[8., -2.], yaw=.3, extent=[1., 2., 4.], **{'class': 'Car'}),
              dict(position=[12., 1.], yaw=0., extent=[1., 2., 4.], **{'class': 'Route'})]
    with tempfile.TemporaryDirectory() as directory:
        previous = os.getcwd()
        try:
            os.chdir(directory)
            for hazard in (False, True):
                agent['create_BEV'](labels, hazard, np.array([[8., 2.]]), np.array([[1., 0.], [3., 1.]]))
                paths = list(Path('bev_viz').glob('*.png'))
                assert len(paths) == 1
                with Image.open(paths[0]) as image:
                    image.load()
                    assert image.size == (310, 310) and image.mode == 'RGB'
                    result['plant_%s' % hazard] = fingerprint(np.asarray(image))
                paths[0].unlink()
        finally:
            os.chdir(previous)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record', type=Path)
    mode.add_argument('--compare', type=Path)
    args = parser.parse_args()
    actual = snapshots()
    if args.record:
        args.record.write_text(json.dumps(dict(pillow=PIL.__version__, numpy=np.__version__,
                                               opencv=cv2.__version__, snapshots=actual), indent=2) + '\n')
    else:
        expected = json.loads(args.compare.read_text())
        assert actual == expected['snapshots'], 'Image output differs from recorded baseline'
        print('Baseline Pillow:', expected['pillow'])
    print('PASS: %d image snapshots; Pillow %s; NumPy %s; OpenCV %s' %
          (len(actual), PIL.__version__, np.__version__, cv2.__version__))
