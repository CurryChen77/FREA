"""Configured Traffic Manager ports must survive actor control transfers.

Execute the original method bodies with actor/provider doubles. These tests do
not start CARLA, import torch, or reserve GPU resources.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def method(path, class_name, name, provider, **globals_):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(CarlaDataProvider=provider, **globals_)
    exec(compile(ast.Module(body=[body], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class TrafficManagerPorts(unittest.TestCase):
    def setUp(self):
        self.provider = Mock()
        self.provider.get_traffic_manager_port.return_value = 22006
        self.actor = Mock()
        self.actor.id = 42

    def test_background_activation_uses_configured_manager(self):
        activate = method('frea/scenario/scenario_definition/route_scenario.py',
                          'RouteScenario', 'activate_background_actors', self.provider)
        far = Mock()
        self.provider.get_location.side_effect = lambda actor: actor.get_location()
        self.actor.get_location.return_value.distance.return_value = 10
        far.get_location.return_value.distance.return_value = 100
        state = SimpleNamespace(ego_vehicle=Mock(), unactivated_actors=[self.actor, far])
        activate(state)
        self.actor.set_autopilot.assert_called_once_with(True, 22006)
        far.set_autopilot.assert_not_called()
        self.assertEqual(state.unactivated_actors, [far])

    def transfer(self, truncated):
        transfer = method('frea/gym_carla/envs/carla_env.py', 'CarlaEnv',
                          '_remove_and_clean_CBV', self.provider)
        state = SimpleNamespace(CBVs={42: self.actor}, CBVs_nearby_vehicles={42: []},
                                ego_vehicle=object(), _remove_CBV_sensor=Mock())
        self.provider.get_CBVs_reach_goal_by_ego.return_value = {42: self.actor}
        transfer(state, dict(CBVs_truncated={42: True} if truncated else {},
                             CBVs_terminated={} if truncated else {42: True}))
        self.actor.set_autopilot.assert_called_once_with(enabled=True, tm_port=22006)
        self.assertFalse(state.CBVs)

    def test_truncated_vehicle_returns_to_configured_manager(self):
        self.transfer(truncated=True)

    def test_goal_vehicle_returns_to_configured_manager(self):
        self.transfer(truncated=False)

    def test_selected_vehicle_leaves_configured_manager(self):
        select = method('frea/gym_carla/envs/carla_env.py', 'CarlaEnv', 'CBVs_selection',
                        self.provider, get_CBV_candidates=Mock(return_value=[self.actor]),
                        find_closest_vehicle=Mock(return_value=self.actor),
                        get_nearby_vehicles=Mock(return_value=[]))
        state = SimpleNamespace(scenario_agent_learnable=True, CBVs={}, time_step=0,
                                ego_vehicle=object(), goal_waypoint=object(), search_radius=25,
                                ego_length=4, CBVs_select_method='rule-based',
                                register_CBV_sensor=Mock(), CBVs_nearby_vehicles={},
                                scenario_manager=Mock())
        select(state)
        self.actor.set_autopilot.assert_called_once_with(enabled=False, tm_port=22006)
        self.assertEqual(state.CBVs, {42: self.actor})


if __name__ == '__main__':
    unittest.main()
