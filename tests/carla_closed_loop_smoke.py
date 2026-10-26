"""Run one bounded, model-free FREA episode on an owned CARLA 0.9.16 server.

Linux only. Configure TMP/XDG/cache paths inside the project before invocation.
The launcher holds common GPU/port flock leases until both child processes exit.
It never connects to or stops an existing simulator. See the security audit for
the invocation used on the 4090D server. No training or checkpoints are involved.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def stop_owned(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def client(args, output):
    # Guard the private entry point against accidentally using another server.
    command = Path('/proc/%d/cmdline' % args.server_pid).read_bytes().split(b'\0')
    assert ('-carla-rpc-port=%d' % args.port).encode() in command
    assert any(b'CarlaUE4-Linux-Shipping' in part for part in command)
    import carla
    import faulthandler
    import imageio.v2 as imageio
    import numpy as np
    import pygame
    import torch
    from PIL import Image
    from frea.carla_runner import CarlaRunner
    from frea.gym_carla.env_wrapper import VectorWrapper
    from frea.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
    from frea.scenario.tools.scenario_utils import scenario_parse
    from frea.util.run_util import load_config, VideoWriter
    from frea.util.torch_util import set_seed, set_torch_variable

    torch.set_num_threads(2)
    set_torch_variable('cpu')
    set_seed(2026)
    CarlaDataProvider.set_random_seed(2026)
    common = dict(ROOT_DIR=str(ROOT), mode='eval', eval_mode='render',
                  eval_obj='ego', num_scenario=1, seed=2026,
                  output_dir=str(output / 'frea'), port=args.port,
                  tm_port=args.port + 6, fixed_delta_seconds=0.1,
                  frame_skip=1, spectator=False, auto_ego=False,
                  viz_route=False, save_video=False, CBV_selection='rule-based',
                  pretrain_ego='expert', pretrain_cbv='standard')
    agent = load_config(str(ROOT / 'frea/agent/config/behavior.yaml'))
    scenario = load_config(str(ROOT / 'frea/scenario/config/standard_eval.yaml'))
    feasibility = load_config(str(ROOT / 'frea/feasibility/config/HJR.yaml'))
    agent.update(common, obs_type='ego_state')
    scenario.update(common, scenario_type='Scenario9_Town05_1x.json', route_id=8)
    feasibility.update(common)
    write_json(output / 'config.json', dict(agent=agent, scenario=scenario,
                                            steps=args.steps, traffic_intensity=0,
                                            display_size=128, enable_sem=True))
    runner = CarlaRunner(agent, scenario, feasibility)
    runner.client.set_timeout(60.0)
    runner.env_params.update(max_episode_step=args.steps, traffic_intensity=0,
                             display_size=128, lidar_bin=60 / 128, enable_sem=True)
    world = None
    video = None
    try:
        assert runner.client.get_server_version() == '0.9.16'
        configs = scenario_parse(scenario, runner.logger)['Town05']
        assert len(configs) == 1
        runner.current_map = 'Town05'
        runner._init_world('Town05')
        world = runner.world
        tm = runner.client.get_trafficmanager(args.port + 6)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(2026)
        runner._init_renderer()
        runner.env = VectorWrapper(runner.env_params, scenario, world,
                                   runner.birdeye_render, runner.display, False,
                                   None, runner.logger)
        print('Resetting FREA environment', flush=True)
        faulthandler.dump_traceback_later(10, repeat=True)
        try:
            obs, infos = runner.env.reset(configs)
        finally:
            faulthandler.cancel_dump_traceback_later()
        print('FREA environment reset complete', flush=True)
        runner.agent_policy.set_mode('eval')
        runner.scenario_policy.set_mode('eval')
        runner.agent_policy.set_ego_and_route(runner.env.get_ego_vehicles(), infos)
        ego = runner.env.get_ego_vehicles()[0]
        env = runner.env.env_list[0].unwrapped
        previous = ego.get_location()
        start_frame = world.get_snapshot().frame
        distance = 0.0
        speeds, rewards, hashes, controls = [], [], set(), []
        video_path = output / 'closed-loop.mp4'
        video = VideoWriter(filename=str(video_path), fps=10)
        for step in range(args.steps):
            actions = runner.agent_policy.get_action(obs, infos, deterministic=True)
            assert np.isfinite(actions).all()
            scenario_actions = runner.scenario_policy.get_action(obs, infos, deterministic=True)
            _, obs, reward, done, _, infos = runner.env.step(
                actions, scenario_actions, {'agent': False, 'scenario': False})
            assert np.isfinite(reward).all()
            assert not env.ego_collide
            assert env.camera_img.shape == (128, 128, 3)
            assert env.sem_img.shape == (128, 128)
            assert env.camera_img.var() > 0
            location = ego.get_location()
            distance += location.distance(previous)
            previous = location
            velocity = ego.get_velocity()
            speeds.append(float(np.linalg.norm([velocity.x, velocity.y, velocity.z])))
            rewards.append(float(reward[0]))
            controls.append(actions[0].tolist())
            hashes.add(hashlib.sha256(env.camera_img.tobytes()).hexdigest())
            video.add(pygame.surfarray.array3d(runner.display).transpose(1, 0, 2))
            print('step=%d frame=%d speed=%.3f distance=%.3f' %
                  (step + 1, world.get_snapshot().frame, speeds[-1], distance), flush=True)
            assert not bool(done[0]) or step == args.steps - 1, 'Episode ended early'
        video.close()
        video = None
        assert distance > 1.0 and max(speeds) > 1.0
        assert len(hashes) > 1, 'RGB sensor did not update'
        from frea.agent.agent_utils.explainability_utils import get_masked_viz_3rd_person
        masked = get_masked_viz_3rd_person(env.BGR_img, env.sem_img)
        for name, array in [('camera.png', env.camera_img), ('masked.png', masked)]:
            Image.fromarray(array).save(output / name)
            with Image.open(output / name) as decoded:
                np.testing.assert_array_equal(np.asarray(decoded), array)
        reader = imageio.get_reader(video_path)
        try:
            frames = sum(1 for _ in reader.iter_data())
        finally:
            reader.close()
        assert frames == args.steps
        result = dict(passed=True, server='0.9.16', seed=2026, steps=args.steps,
                      simulated_seconds=args.steps * 0.1, distance_m=distance,
                      max_speed_mps=max(speeds), unique_rgb_images=len(hashes),
                      frame_delta=world.get_snapshot().frame - start_frame,
                      video_frames=frames, rewards=rewards, controls=controls,
                      collided=False, checkpoints_loaded=0, training_calls=0)
    finally:
        if video is not None:
            video.close()
        # Only this launcher's server can be reached; clean even a partial reset.
        if world is not None:
            if runner.env is not None:
                runner.env.clean_up()
            CarlaDataProvider.clean_up()
            for actor in world.get_actors():
                if actor.type_id.startswith(('vehicle.', 'sensor.')):
                    if actor.type_id.startswith('sensor.'):
                        actor.stop()
                    actor.destroy()
            world.tick()
            remaining = [a.id for a in world.get_actors()
                         if a.type_id.startswith(('vehicle.', 'sensor.'))]
            assert not remaining, 'Owned simulator actor cleanup incomplete'
        pygame.quit()
    result['remaining_actors'] = 0
    write_json(output / 'result.json', result)
    print('PASS: FREA closed-loop smoke', flush=True)


def launch(args, output):
    coordination = Path('/home/keyuchen/.codex/coordination')
    ports = [args.port, args.port + 1, args.port + 2, args.port + 6]
    server = process = None
    server_pid = None
    with ExitStack() as stack:
        for path in [coordination / ('unicraft-gpu-%d.lock' % args.gpu)] + [
                coordination / ('unicraft-port-%d.lock' % p) for p in ports]:
            lease = stack.enter_context(path.open('r'))
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                          '--format=csv,noheader,nounits'], text=True)
        selected = [line.split(', ') for line in before.splitlines()
                    if line.split(',')[0].strip() == str(args.gpu)][0]
        assert int(selected[2]) < 256, 'Selected GPU is in use'
        for port in ports:
            for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
                with socket.socket(socket.AF_INET, kind) as probe:
                    probe.bind(('0.0.0.0', port))
        for folder in ('home', 'cache', 'tmp'):
            (output / folder).mkdir()
        environment = dict(os.environ, HOME=str(output / 'home'),
                           CUDA_VISIBLE_DEVICES=str(args.gpu),
                           VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',
                           __GL_SHADER_DISK_CACHE_PATH=str(output / 'cache'),
                           __GL_SHADER_DISK_CACHE_SIZE='67108864')
        environment.pop('SDL_VIDEODRIVER', None)
        environment.pop('DISPLAY', None)
        cpus = ','.join(map(str, sorted(os.sched_getaffinity(0))[-4:]))
        binary = args.carla_root / 'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
        command = ['taskset', '-c', cpus, 'nice', '-n', '10', str(binary), 'CarlaUE4',
                   '-RenderOffScreen', '-nosound', '-quality-level=' + args.quality,
                   '-ResX=128', '-ResY=128', '-fps=10', '-unattended',
                   '-carla-rpc-port=%d' % args.port,
                   '-carla-streaming-port=%d' % (args.port + 1),
                   '-carla-secondary-port=%d' % (args.port + 2),
                   '-graphicsadapter=%d' % args.gpu, '-carla-worker-threads=2',
                   '-SaveToUserDir', '-userdir=' + str(output / 'home/unreal'),
                   '-abslog=' + str(output / 'server.log')]
        if args.debugger:
            command[6:6] = ['gdb', '--batch', '-ex', 'set pagination off',
                            '-ex', 'set print thread-events off',
                            '-ex', 'set startup-with-shell off',
                            '-ex', 'set disable-randomization off',
                            '-ex', 'handle SIGPIPE nostop noprint pass',
                            '-ex', 'run', '-ex', 'bt 30', '--args']
        write_json(output / 'launch.json', dict(command=command, gpu=selected,
                                               ports=ports, cpus=cpus,
                                               source_commit=subprocess.check_output(
                                                   ['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                               gpu_before=before))
        try:
            log = stack.enter_context((output / 'server-stdout.log').open('w'))
            server = subprocess.Popen(command, cwd=output, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            import carla
            ready = False
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert server.poll() is None, 'Owned CARLA server exited'
                gpu_processes = subprocess.check_output([
                    'nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                    '--format=csv,noheader'], text=True)
                candidates = {str(server.pid)}
                if args.debugger:
                    candidates.update(subprocess.run(
                        ['ps', '-o', 'pid=', '--ppid', str(server.pid)],
                        text=True, capture_output=True, check=False).stdout.split())
                own = [line for line in gpu_processes.splitlines()
                       if line.split(',')[0].strip() in candidates]
                if own:
                    assert selected[1] in own[0], 'Vulkan selected an unleased GPU'
                    server_pid = int(own[0].split(',')[0])
                try:
                    # A client created before RPC startup can retain a failed
                    # connection; use a fresh probe for each bounded retry.
                    rpc = carla.Client('127.0.0.1', args.port)
                    rpc.set_timeout(1.0)
                    ready = rpc.get_server_version() == '0.9.16' and bool(own)
                except RuntimeError:
                    pass
                if ready:
                    print('Owned CARLA 0.9.16 ready on leased GPU', args.gpu, flush=True)
                    break
                time.sleep(1)
            assert ready, 'CARLA did not become ready within 90 seconds'
            child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            client_log = stack.enter_context((output / 'client.log').open('w'))
            client_command = [sys.executable, '-u', str(Path(__file__).resolve()),
                              '--client', '--server-pid', str(server_pid),
                              '--port', str(args.port), '--steps', str(args.steps),
                              '--output-dir', str(output)]
            process = subprocess.Popen(client_command, cwd=ROOT, env=child_env,
                                       stdout=client_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            assert process.wait(timeout=180) == 0, 'FREA client failed; see client.log'
        finally:
            stop_owned(process)
            if args.debugger and server_pid and Path('/proc/%d/cmdline' % server_pid).exists():
                command_line = Path('/proc/%d/cmdline' % server_pid).read_bytes()
                if ('-carla-rpc-port=%d' % args.port).encode() in command_line:
                    os.kill(server_pid, signal.SIGTERM)
            stop_owned(server)
            write_json(output / 'process-cleanup.json', dict(
                server_pid=server.pid if server else None,
                server_returncode=server.poll() if server else None,
                client_returncode=process.poll() if process else None,
                leases_released_after_exit=True))
        assert (output / 'result.json').is_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--carla-root', type=Path)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--port', type=int, default=22000)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--quality', choices=('Low', 'Epic'), default='Epic')
    parser.add_argument('--debugger', action='store_true')
    parser.add_argument('--client', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--server-pid', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    assert ROOT / 'runs' in output.parents
    assert 30 <= args.steps <= 100
    if args.client:
        client(args, output)
    else:
        assert args.carla_root and (args.carla_root / 'VERSION').read_text().strip() == '0.9.16'
        output.mkdir(parents=True, exist_ok=False)
        launch(args, output)
