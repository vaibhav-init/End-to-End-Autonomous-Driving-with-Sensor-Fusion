from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from radar.cshenron_core import SEMANTIC_LIDAR_DTYPE
from radar.front_radar import CShenronFrontRadar


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeActor:
    def __init__(self, actor_id, location, velocity):
        self.id = actor_id
        self._location = location
        self._velocity = velocity
        self.is_alive = True

    def get_location(self):
        return self._location

    def get_velocity(self):
        return self._velocity

    def get_transform(self):
        return SimpleNamespace(rotation=SimpleNamespace(yaw=0.0))


class FakeBlueprint:
    def __init__(self):
        self.attributes = {}

    def has_attribute(self, _name):
        return True

    def set_attribute(self, name, value):
        self.attributes[name] = value


class FakeSensor:
    def __init__(self):
        self.is_alive = True
        self.callback = None

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        pass

    def destroy(self):
        self.is_alive = False


class FakeWorld:
    def __init__(self, actors):
        self.actors = {actor.id: actor for actor in actors}
        self.blueprint = FakeBlueprint()
        self.sensor = FakeSensor()

    def get_blueprint_library(self):
        return SimpleNamespace(find=lambda _name: self.blueprint)

    def spawn_actor(self, _blueprint, _transform, attach_to=None):
        return self.sensor

    def get_actor(self, actor_id):
        return self.actors.get(actor_id)


class FrontRadarTest(unittest.TestCase):
    def test_cshenron_adapter_preserves_scalar_contract_and_closing_speed(self):
        ego = FakeActor(1, FakeVector(0.0, 0.0, 0.0), FakeVector(10.0, 0.0, 0.0))
        lead = FakeActor(77, FakeVector(25.0, 0.0, 0.0), FakeVector(5.0, 0.0, 0.0))
        world = FakeWorld((ego, lead))
        fake_carla = SimpleNamespace(
            Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
        )

        with patch("radar.front_radar._carla_module", return_value=fake_carla):
            radar = CShenronFrontRadar(ego, world, range_m=100.0, fps=20)

        returns = np.zeros(3, dtype=SEMANTIC_LIDAR_DTYPE)
        returns[0] = (25.0, -0.2, 0.0, 1.0, 77, 14)
        returns[1] = (25.0, 0.0, 0.1, 1.0, 77, 14)
        returns[2] = (25.0, 0.2, -0.1, 1.0, 77, 14)
        measurement = SimpleNamespace(raw_data=returns.tobytes(), frame=123)
        radar._on_semantic_lidar(measurement)
        radar.update_ego_speed(10.0)

        state = radar.get()
        self.assertEqual(
            set(state), {"distance", "relative_velocity", "obstacle_speed"}
        )
        self.assertAlmostEqual(state["distance"], 25.0, delta=0.7)
        self.assertAlmostEqual(state["relative_velocity"], 5.0, delta=0.5)
        self.assertAlmostEqual(state["obstacle_speed"], 5.0, delta=0.5)
        self.assertEqual(radar.diagnostics()["target_object_id"], 77)
        radar.cleanup()


if __name__ == "__main__":
    unittest.main()
