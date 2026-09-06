from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from radar.cshenron_core import SEMANTIC_LIDAR_DTYPE
from radar.front_radar import CShenronFrontRadar, RealisticFrontRadar
from radar.realistic_core import load_realistic_radar_config


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

    def get_weather(self):
        return SimpleNamespace(
            precipitation=0.0,
            wetness=0.0,
            fog_density=0.0,
            dust_storm=0.0,
        )


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
            radar = CShenronFrontRadar(
                ego,
                world,
                range_m=100.0,
                fps=20,
                capture_debug=True,
            )

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
        debug = radar.debug_snapshot()
        self.assertEqual(debug["ideal_targets"][0]["object_id"], 77)
        self.assertEqual(debug["semantic_tag_counts"]["14"]["name"], "Car")
        radar.cleanup()

    def test_sensor_callback_never_calls_into_the_simulator(self):
        """Regression: an RPC inside the LiDAR callback deadlocked sync mode.

        ``_read_environment`` called ``world.get_weather()`` from inside
        ``_on_semantic_lidar``. In a synchronous run the main thread is inside
        ``world.tick()`` waiting for the sensor to deliver, while the sensor
        thread waits for a server that cannot answer until the tick finishes.
        Collection hung on ``world.tick()`` with no traceback and no CSV.

        The weather must therefore be read on the main thread only. This test
        makes any simulator call from the callback an immediate failure.
        """

        ego = FakeActor(1, FakeVector(), FakeVector(10.0, 0.0, 0.0))
        lead = FakeActor(77, FakeVector(25.0, 0.0, 0.0), FakeVector(5.0, 0.0, 0.0))
        world = FakeWorld((ego, lead))
        fake_carla = SimpleNamespace(
            Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda *args, **kwargs: SimpleNamespace(
                args=args, kwargs=kwargs
            ),
        )
        # Deterministic profile: one scan is enough to confirm a track, so a
        # failed callback shows up as the empty state rather than as noise.
        config = load_realistic_radar_config("ideal_target_list_v1")

        with patch("radar.front_radar._carla_module", return_value=fake_carla):
            radar = RealisticFrontRadar(
                ego, world, range_m=100.0, fps=20, config=config
            )

        # The main-thread hook is allowed to read weather, and must, or the
        # model would run the whole session on a default environment.
        weather_calls = []
        real_get_weather = world.get_weather
        world.get_weather = lambda: (
            weather_calls.append(1) or real_get_weather()
        )
        radar.update_ego_speed(10.0)
        self.assertEqual(len(weather_calls), 1)

        returns = np.zeros(3, dtype=SEMANTIC_LIDAR_DTYPE)
        returns[0] = (25.0, -0.2, 0.0, 1.0, 77, 14)
        returns[1] = (25.0, 0.0, 0.1, 1.0, 77, 14)
        returns[2] = (25.0, 0.2, -0.1, 1.0, 77, 14)
        measurement = SimpleNamespace(
            raw_data=returns.tobytes(), frame=123, timestamp=6.15
        )

        def deadlock(*_args, **_kwargs):
            raise AssertionError(
                "sensor callback read the weather over RPC; in a synchronous "
                "run this deadlocks world.tick()"
            )

        world.get_weather = deadlock
        radar._on_semantic_lidar(measurement)

        # The callback must have completed and selected the lead, not fallen
        # back to the empty state -- the adapter swallows callback exceptions,
        # so asserting on the result is what makes this test real.
        self.assertAlmostEqual(radar.get()["distance"], 25.0, delta=1.0)
        self.assertIsNone(radar.diagnostics()["last_error"])

    def test_static_snr_offset_leaves_road_users_alone(self):
        ego = FakeActor(1, FakeVector(), FakeVector(10.0, 0.0, 0.0))
        lead = FakeActor(77, FakeVector(25.0, 0.0, 0.0), FakeVector(5.0, 0.0, 0.0))
        fake_carla = SimpleNamespace(
            Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
        )
        returns = np.zeros(6, dtype=SEMANTIC_LIDAR_DTYPE)
        returns[0] = (25.0, -0.2, 0.0, 1.0, 77, 14)
        returns[1] = (25.0, 0.0, 0.1, 1.0, 77, 14)
        returns[2] = (25.0, 0.2, -0.1, 1.0, 77, 14)
        # A guardrail (tag 28) at 40 m, off the ego path.
        returns[3] = (40.0, 6.0, 0.0, 1.0, 0, 28)
        returns[4] = (40.0, 6.2, 0.1, 1.0, 0, 28)
        returns[5] = (40.0, 6.4, -0.1, 1.0, 0, 28)
        measurement = SimpleNamespace(raw_data=returns.tobytes(), frame=5, timestamp=1.0)
        snr = {}
        for offset in (0.0, 30.0):
            config = load_realistic_radar_config(
                "ideal_target_list_v1",
                overrides={"static_snr_offset_db": offset, "emit_extended_points": False},
            )
            with patch("radar.front_radar._carla_module", return_value=fake_carla):
                radar = RealisticFrontRadar(
                    ego, FakeWorld((ego, lead)), range_m=100.0, fps=20,
                    config=config, capture_debug=True,
                )
            radar.update_ego_speed(10.0)
            radar._on_semantic_lidar(measurement)
            ideal = radar.debug_snapshot()["ideal_targets"]
            snr[offset] = {}
            for target in ideal:
                tag = int(target["semantic_tag"])
                snr[offset][tag] = max(snr[offset].get(tag, -1.0e9), float(target["snr_db"]))
            radar.cleanup()
        self.assertAlmostEqual(snr[30.0][28] - snr[0.0][28], 30.0, places=6)
        self.assertAlmostEqual(snr[30.0][14] - snr[0.0][14], 0.0, places=6)

    def test_road_user_snr_offset_reaches_the_target_list(self):
        ego = FakeActor(1, FakeVector(), FakeVector(10.0, 0.0, 0.0))
        lead = FakeActor(77, FakeVector(25.0, 0.0, 0.0), FakeVector(5.0, 0.0, 0.0))
        fake_carla = SimpleNamespace(
            Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda *args, **kwargs: SimpleNamespace(args=args, kwargs=kwargs),
        )
        returns = np.zeros(3, dtype=SEMANTIC_LIDAR_DTYPE)
        returns[0] = (25.0, -0.2, 0.0, 1.0, 77, 14)
        returns[1] = (25.0, 0.0, 0.1, 1.0, 77, 14)
        returns[2] = (25.0, 0.2, -0.1, 1.0, 77, 14)
        measurement = SimpleNamespace(raw_data=returns.tobytes(), frame=5, timestamp=1.0)
        snr = {}
        for offset in (0.0, -20.0):
            config = load_realistic_radar_config(
                "ideal_target_list_v1",
                overrides={"road_user_snr_offset_db": offset, "emit_extended_points": False},
            )
            with patch("radar.front_radar._carla_module", return_value=fake_carla):
                radar = RealisticFrontRadar(
                    ego, FakeWorld((ego, lead)), range_m=100.0, fps=20,
                    config=config, capture_debug=True,
                )
            radar.update_ego_speed(10.0)
            radar._on_semantic_lidar(measurement)
            snr[offset] = radar.debug_snapshot()["delivered_detections"][0]["snr_db"]
            radar.cleanup()
        self.assertAlmostEqual(snr[0.0] - snr[-20.0], 20.0, places=6)

    def test_realistic_adapter_preserves_contract_without_carla_install(self):
        ego = FakeActor(1, FakeVector(), FakeVector(10.0, 0.0, 0.0))
        lead = FakeActor(
            77,
            FakeVector(25.0, 0.0, 0.0),
            FakeVector(5.0, 0.0, 0.0),
        )
        world = FakeWorld((ego, lead))
        fake_carla = SimpleNamespace(
            Location=lambda **kwargs: SimpleNamespace(**kwargs),
            Transform=lambda *args, **kwargs: SimpleNamespace(
                args=args,
                kwargs=kwargs,
            ),
        )
        config = load_realistic_radar_config("ideal_target_list_v1")

        with patch("radar.front_radar._carla_module", return_value=fake_carla):
            radar = RealisticFrontRadar(
                ego,
                world,
                range_m=100.0,
                fps=20,
                config=config,
                capture_debug=True,
            )

        returns = np.zeros(3, dtype=SEMANTIC_LIDAR_DTYPE)
        returns[0] = (25.0, -0.2, 0.0, 1.0, 77, 14)
        returns[1] = (25.0, 0.0, 0.1, 1.0, 77, 14)
        returns[2] = (25.0, 0.2, -0.1, 1.0, 77, 14)
        measurement = SimpleNamespace(
            raw_data=returns.tobytes(),
            frame=123,
            timestamp=6.15,
        )
        radar.update_ego_speed(10.0)
        radar._on_semantic_lidar(measurement)

        state = radar.get()
        self.assertEqual(
            set(state),
            {"distance", "relative_velocity", "obstacle_speed"},
        )
        self.assertAlmostEqual(state["distance"], 25.0, delta=0.1)
        self.assertAlmostEqual(state["relative_velocity"], 5.0, delta=0.1)
        self.assertAlmostEqual(state["obstacle_speed"], 5.0, delta=0.1)
        diagnostics = radar.diagnostics()
        self.assertEqual(diagnostics["backend"], "realistic")
        self.assertEqual(diagnostics["selected_truth_object_id"], 77)
        self.assertEqual(diagnostics["selected_semantic_tag"], 14)
        debug = radar.debug_snapshot()
        self.assertEqual(debug["selected"]["truth_object_id"], 77)
        self.assertEqual(debug["delivered_detections"][0]["semantic_tag"], 14)
        radar.cleanup()


if __name__ == "__main__":
    unittest.main()
