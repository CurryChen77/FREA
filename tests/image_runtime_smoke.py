"""CPU integration smoke for the image paths used by FREA.

Install requirements.txt and the matching CARLA Python client. Add CARLA's
PythonAPI/carla directory to PYTHONPATH. Set SDL_VIDEODRIVER=dummy and set
TMPDIR/TMP/TEMP plus MPLCONFIGDIR to directories inside this project.
No CARLA server, model downloads, checkpoints or GPU are required.
"""
import importlib
import io
from pathlib import Path
import tempfile

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    for name in (
        'frea.agent.agent_utils.viz_tokens_bev',
        'frea.agent.agent_utils.explainability_utils',
        'frea.agent.PlanT.PlanT_agent',
        'frea.agent.expert.nav_planner',
        'frea.gym_carla.envs.carla_env',
        'frea.util.run_util',
    ):
        importlib.import_module(name)
        print('Imported', name)

    pixels = np.random.RandomState(2026).randint(0, 256, (48, 64, 3)).astype(np.uint8)
    # Exercise the imageio/Pillow bridge and native codecs with valid local data.
    for fmt in ('PNG', 'JPEG', 'TIFF', 'GIF'):
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format=fmt)
        buffer.seek(0)
        decoded = imageio.imread(buffer, format=fmt)
        assert decoded.shape[:2] == pixels.shape[:2]
        if fmt in ('PNG', 'TIFF'):
            np.testing.assert_array_equal(decoded, pixels)

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(pixels)
        fig.savefig(directory / 'plot.png')
        plt.close(fig)
        with Image.open(directory / 'plot.png') as rendered:
            rendered.load()
            assert rendered.size == (200, 200)

        from frea.util.run_util import VideoWriter
        video = directory / 'frames.mp4'
        with VideoWriter(filename=str(video), fps=5) as writer:
            for _ in range(3):
                writer.add(pixels)
        # VideoWriter suppresses write errors, so verify decoded frame content.
        reader = imageio.get_reader(video)
        try:
            frames = list(reader.iter_data())
            assert len(frames) == 3 and frames[0].shape == pixels.shape
            assert np.abs(frames[0].astype(float) - pixels).mean() < 60
        finally:
            reader.close()
    print('PASS: full image-module imports, codecs, plotting and FREA video export')


if __name__ == '__main__':
    main()
