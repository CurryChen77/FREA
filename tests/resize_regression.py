"""Compare camera/display resizing against a legacy scikit-image 0.19.3 fixture.

Set TMPDIR/TMP/TEMP inside the project. Record only in the legacy environment;
compare in the candidate environment. This does not start a CARLA server.
"""
import argparse
from pathlib import Path

import numpy as np
import pygame
import skimage
import skimage.transform

from pillow_regression import helpers


def outputs():
    namespace = helpers('frea/gym_carla/envs/misc.py',
                        ['display_to_rgb', 'rgb_to_display_surface'])
    namespace.update(skimage=skimage, pygame=pygame)
    random = np.random.RandomState(2026)
    camera = random.randint(0, 256, (48, 64, 3)).astype(np.uint8)
    result = {}
    pygame.display.init()
    try:
        pygame.display.set_mode((1, 1))
        for size in (16, 96):
            result['display_%d' % size] = namespace['display_to_rgb'](camera, size)
            # Matches the CARLA camera observation conversion, including *255.
            result['camera_%d' % size] = skimage.transform.resize(camera, (size, size)) * 255
            surface = namespace['rgb_to_display_surface'](camera, size)
            result['surface_%d' % size] = pygame.surfarray.array3d(surface)
    finally:
        pygame.display.quit()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record', type=Path)
    mode.add_argument('--compare', type=Path)
    args = parser.parse_args()
    actual = outputs()
    if args.record:
        if skimage.__version__ != '0.19.3':
            raise RuntimeError('Record the fixture with scikit-image 0.19.3')
        np.savez_compressed(args.record, **actual)
    else:
        with np.load(args.compare, allow_pickle=False) as expected:
            assert set(expected.files) == set(actual)
            for name, array in actual.items():
                assert array.dtype == expected[name].dtype
                np.testing.assert_allclose(array, expected[name], rtol=0, atol=1e-12,
                                           err_msg=name)
    print('PASS: %d camera/display arrays; scikit-image %s; NumPy %s' %
          (len(actual), skimage.__version__, np.__version__))
