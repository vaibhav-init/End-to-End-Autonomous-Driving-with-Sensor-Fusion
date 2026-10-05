"""Radar-to-radar interference (RRI) for the realistic sensor model.

Why this is a separate module
-----------------------------
The interference already in ``realistic_core`` is a two-state Markov burst:
``interference_enter_probability`` flips a flag, detection probability is
multiplied by ``interference_detection_scale`` and the clutter rate by
``interference_clutter_multiplier``. That models *"the link budget got worse
for a while"*. It has no notion of a second radar, so it cannot produce the
artifacts that actually break a downstream tracker.

RRI is a coherent signal-level effect. When radar *j* transmits in the same
band as radar *i*, some of *j*'s energy leaks into *i*'s receiver through
finite antenna-to-antenna isolation. The consequences that matter at the
target-list level are:

1. **Phantom targets** at the apparent position of the interfering radar.
   The leakage is a real signal with a real delay, so after the matched
   filter it lands in a range/azimuth/Doppler cell and CFAR reports it as a
   target. There is nothing there. This is the artifact that causes phantom
   braking.
2. **Range-ambiguity replicas.** Chirp-to-chirp phase alignment puts
   additional phantoms at ``|k * R_unamb - d|`` where
   ``R_unamb = c * T_chirp / 2``. For a 77 GHz FMCW radar with a 1 us chirp
   period, ``R_unamb`` is about 150 m, so a 30 m interferer also paints a
   phantom near 120 m.
3. **Desensitisation.** The interference adds to the receiver noise, so real
   targets lose SNR: ``SINR = 1 / (1/SNR + 1/INR)``.
4. **False-alarm inflation.** A raised noise floor drags the CFAR threshold
   down. In the linear region of the detector the false-alarm count grows
   roughly in proportion to the interference-to-noise ratio.

What is deliberately not modelled
---------------------------------
No waveform superposition, no antenna patterns beyond a scalar isolation
figure, no Doppler coupling between the two radars' chirp slopes, and no
phase. Those need a waveform simulator (``radar/rri_waveform.py`` is not
written; see the roadmap note in ``RRI_SCOPE.md``). What is here is a
structured, parameterised surrogate: the artifact *positions* and *relative
amplitudes* follow from the physics, but the fine structure does not. It is
the same class of model as ``geometry`` multipath in ``multipath.py`` --
analytic geometry feeding a fast target-list model -- not a waveform model.

Sign and unit conventions match the rest of ``carla4/radar``:

* x forward, y right; azimuth from boresight, positive right.
* ``relative_velocity_mps`` positive = closing (approaching).
* All powers in dBm, all losses/gains in dB, all distances in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Speed of light, m/s.
SPEED_OF_LIGHT_MPS = 299_792_458.0

# Minimum carrier used for the free-space path loss between two radars.
# Only matters through ``log10(f)``, so a single representative automotive
# band frequency is enough; it is a parameter on the profile, not a constant.
DEFAULT_CARRIER_FREQUENCY_HZ = 77.0e9


@dataclass(frozen=True)
class InterferingRadar:
    """One other radar that is coupling into the sensor under test.

    The geometry is expressed exactly like an ``IdealRadarTarget`` -- range,
    azimuth and radial velocity in the ego frame -- because that is the frame
    the leakage artifact appears in. ``boresight_alignment`` is the cosine
    between the interferer's antenna boresight and the line joining the two
    radars; it scales the coupling because a radar looking away from us leaks
    far less than one looking at us.
    """

    object_id: int
    distance_m: float
    azimuth_rad: float
    relative_velocity_mps: float = 0.0
    transmit_power_dbm: float = 10.0
    antenna_gain_dbi: float = 8.0
    boresight_alignment: float = 1.0
    label: str = ""

    def clamped(self) -> "InterferingRadar":
        """Return a copy with unusable values replaced by safe ones."""

        return InterferingRadar(
            object_id=int(self.object_id),
            distance_m=float(max(0.1, self.distance_m)),
            azimuth_rad=float(self.azimuth_rad),
            relative_velocity_mps=float(self.relative_velocity_mps),
            transmit_power_dbm=float(self.transmit_power_dbm),
            antenna_gain_dbi=float(self.antenna_gain_dbi),
            boresight_alignment=float(
                np.clip(self.boresight_alignment, 0.0, 1.0)
            ),
            label=str(self.label),
        )


@dataclass(frozen=True)
class RRIParameters:
    """Sensor-side waveform and coupling parameters for the RRI model.

    These are properties of *our* radar plus the installation, not of the
    interfering radar, so they live on ``RealisticRadarConfig`` rather than
    being passed per-interferer.
    """

    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ
    noise_figure_db: float = 4.0
    noise_bandwidth_hz: float = 1.0e9
    chirp_period_s: float = 1.0e-6
    # Antenna-to-antenna isolation when boresights are aligned. Real mounts
    # sit between 30 and 60 dB apart; the boresight term below is relative to
    # this figure, not an absolute.
    antenna_isolation_db: float = 40.0
    # Our own coherent FFT gain, in chirps. This is what suppresses a
    # phase-locked interferer, so it belongs to the victim, not the interferer.
    coherent_processing_chirps: int = 64
    # How much of that gain the interference actually earns. Two
    # unsynchronised radars in the same band only stay locked if their carrier
    # offsets match, which is the exceptional case, so the default is well
    # below 1: measured RRI is not the fully coherent worst case.
    interference_coherence: float = 0.3
    max_ghost_orders: int = 2
    # A phantom is only reported when the interference exceeds the noise by
    # this margin after processing gain. Below it, CFAR does not cross
    # threshold and nothing is reported.
    min_inr_db_for_phantom: float = 3.0
    # Extra SNR loss applied to real targets when a phantom is present, on top
    # of the SINR combination. Models the threshold drag and the receiver
    # loading that a strong interferer causes.
    desensitisation_db: float = 1.0
    # False-alarm count multiplier per dB of interference-to-noise ratio
    # above the noise floor, saturating at ``max_false_alarm_multiplier``.
    false_alarm_slope: float = 0.35
    max_false_alarm_multiplier: float = 25.0
    # Probability that a phantom survives CFAR, per dB of INR, before
    # clamping. Keeps phantoms intermittent like real CFAR detections instead
    # of guaranteed.
    phantom_detection_slope: float = 0.20
    max_phantom_probability: float = 0.95

    @property
    def unambiguous_range_m(self) -> float:
        """Maximum unambiguous range from the chirp period, ``c * T / 2``."""

        return 0.5 * SPEED_OF_LIGHT_MPS * max(self.chirp_period_s, 1.0e-9)

    @property
    def noise_power_dbm(self) -> float:
        """Thermal noise floor in the configured bandwidth."""

        bandwidth_hz = max(self.noise_bandwidth_hz, 1.0)
        return (
            -174.0
            + 10.0 * math.log10(bandwidth_hz)
            + float(self.noise_figure_db)
        )

    def validated(self) -> "RRIParameters":
        """Return a copy with every field forced into a usable range."""

        return RRIParameters(
            carrier_frequency_hz=max(1.0e6, float(self.carrier_frequency_hz)),
            noise_figure_db=float(max(0.0, self.noise_figure_db)),
            noise_bandwidth_hz=max(1.0, float(self.noise_bandwidth_hz)),
            chirp_period_s=float(
                np.clip(self.chirp_period_s, 1.0e-9, 1.0e-2)
            ),
            antenna_isolation_db=float(self.antenna_isolation_db),
            coherent_processing_chirps=max(
                1, int(self.coherent_processing_chirps)
            ),
            interference_coherence=float(
                np.clip(self.interference_coherence, 0.0, 1.0)
            ),
            max_ghost_orders=int(max(0, self.max_ghost_orders)),
            min_inr_db_for_phantom=float(self.min_inr_db_for_phantom),
            desensitisation_db=max(0.0, float(self.desensitisation_db)),
            false_alarm_slope=max(0.0, float(self.false_alarm_slope)),
            max_false_alarm_multiplier=max(
                1.0, float(self.max_false_alarm_multiplier)
            ),
            phantom_detection_slope=max(
                0.0, float(self.phantom_detection_slope)
            ),
            max_phantom_probability=float(
                np.clip(self.max_phantom_probability, 0.0, 1.0)
            ),
        )


def free_space_path_loss_db(
    distance_m: float,
    frequency_hz: float,
) -> float:
    """Free-space path loss in dB between two antennas."""

    distance_m = max(float(distance_m), 0.1)
    frequency_hz = max(float(frequency_hz), 1.0e6)
    return 20.0 * math.log10(
        4.0
        * math.pi
        * distance_m
        * frequency_hz
        / SPEED_OF_LIGHT_MPS
    )


def interference_power_dbm(
    interferer: InterferingRadar,
    params: RRIParameters,
) -> float:
    """Power from one interfering radar arriving at our receiver, in dBm.

    Link budget from the interferer to us: transmit power plus its antenna
    gain, minus free-space spreading, minus the antenna-to-antenna isolation.
    The boresight term is a relative pattern loss: an interferer pointed at us
    gets the quoted ``antenna_isolation_db``, and one pointing away leaks
    better according to a cosine-power pattern, floored so a fully
    side-on interferer still leaks 30 dB.
    """

    alignment = float(np.clip(interferer.boresight_alignment, 0.0, 1.0))
    # Cosine-power pattern as a positive loss in dB: an interferer pointed at
    # us gets the quoted antenna_isolation_db, and one pointing away leaks
    # better by up to 30 dB. Clipped at the low end so a fully side-on
    # interferer still gets 30 dB of extra isolation rather than an unbounded
    # gain from a degenerate alignment.
    pattern_loss_db = -10.0 * math.log10(max(alignment, 1.0e-3))
    pattern_loss_db = float(np.clip(pattern_loss_db, 0.0, 30.0))
    return (
        float(interferer.transmit_power_dbm)
        + float(interferer.antenna_gain_dbi)
        - free_space_path_loss_db(interferer.distance_m, params.carrier_frequency_hz)
        - float(params.antenna_isolation_db)
        - pattern_loss_db
    )


def rejection_db(params: RRIParameters) -> float:
    """Gain against a coherent interferer from *our* coherent processing.

    A target that stays phase-locked across our chirp sequence is cancelled by
    coherent integration of ``N`` chirps at ``10 log10(N)``. Two unsynchronised
    radars in the same band do not stay locked unless their carrier offsets
    happen to be equal, so the achievable rejection is scaled by
    ``interference_coherence``: 1.0 means identical carriers (full gain, the
    worst case for us), lower values mean the beat drifts across the sequence
    and averaging leaks less of the interferer through.

    The gain depends on the *victim's* processing, not the interferer's chirp
    count, which is why this takes no interferer argument.
    """

    chirps = max(1, int(params.coherent_processing_chirps))
    coherent = float(np.clip(params.interference_coherence, 0.0, 1.0))
    return 10.0 * math.log10(chirps) * coherent


def interference_to_noise_db(
    interferer: InterferingRadar,
    params: RRIParameters,
) -> float:
    """Interference-to-noise ratio at our receiver, in dB, after processing.

    Positive values mean the interference sits above our own noise floor and
    can therefore raise the CFAR threshold.
    """

    received_dbm = interference_power_dbm(interferer, params)
    return received_dbm - params.noise_power_dbm - rejection_db(params)


def desensitise_snr_db(snr_db: float, max_inr_db: float) -> float:
    """Combine a target's SNR with interference power.

    Returns the SINR in dB. With no interference the input is returned
    unchanged; as ``INR`` grows the output falls towards the interference
    floor rather than dropping to zero, which is the physically correct limit
    -- a target buried under a strong interferer sits at the interference
    level, it does not vanish.

    ``SINR = S / (N + I)``, i.e. ``SNR / (1 + INR)`` in linear terms. Note this
    is *not* ``1 / (1/SNR + 1/INR)``: that form treats both quantities as
    inverses and returns the target's own SNR whenever the interference sits
    below it, which is the opposite of what interference does.
    """

    if not math.isfinite(snr_db):
        return float(snr_db)
    if max_inr_db <= -300.0:
        return float(snr_db)
    signal_linear = 10.0 ** (max(snr_db, -200.0) / 10.0)
    interference_linear = 10.0 ** (max(max_inr_db, -300.0) / 10.0)
    sinr_linear = signal_linear / (1.0 + interference_linear)
    return float(10.0 * math.log10(max(sinr_linear, 1.0e-30)))


def false_alarm_multiplier(max_inr_db: float, params: RRIParameters) -> float:
    """Factor to apply to the false-alarm rate for a given INR.

    Linear in dB above the floor and saturating, so a very strong interferer
    cannot make the false-alarm rate unbounded.
    """

    excess_db = max(0.0, float(max_inr_db))
    multiplier = 1.0 + float(params.false_alarm_slope) * excess_db
    return float(min(multiplier, params.max_false_alarm_multiplier))


def phantom_probability(max_inr_db: float, params: RRIParameters) -> float:
    """Probability that a phantom survives CFAR and is reported."""

    excess_db = max(
        0.0,
        float(max_inr_db) - float(params.min_inr_db_for_phantom),
    )
    probability = 1.0 - math.exp(
        -float(params.phantom_detection_slope) * excess_db
    )
    return float(
        np.clip(probability, 0.0, params.max_phantom_probability)
    )


@dataclass(frozen=True)
class RRIReport:
    """Outcome of evaluating one scan's worth of interfering radars."""

    max_inr_db: float = 0.0
    total_inr_db: float = 0.0
    desensitisation_db: float = 0.0
    false_alarm_multiplier: float = 1.0
    phantom_count: int = 0
    phantoms: tuple = ()
    interferer_count: int = 0

    @property
    def active(self) -> bool:
        """True when any interference is strong enough to change the picture."""

        return self.max_inr_db > 0.0


def evaluate_rri(
    interferers,
    params: RRIParameters,
    rng=None,
) -> RRIReport:
    """Evaluate interference from a list of ``InterferingRadar`` for one scan.

    Returns the desensitisation to apply to real targets, the false-alarm rate
    multiplier, and the phantom targets to inject as detections. Phantoms are
    drawn at the interferer's apparent position plus its range-ambiguity
    replicas, and each is reported with its own detection probability so
    phantoms flicker the way real CFAR detections do rather than appearing on
    every scan.
    """

    params = params.validated()
    usable = [item.clamped() for item in (interferers or ())]
    usable = [
        item
        for item in usable
        if math.isfinite(item.distance_m)
        and math.isfinite(item.azimuth_rad)
        and math.isfinite(item.relative_velocity_mps)
    ]
    if not usable:
        return RRIReport()

    rng = rng if rng is not None else np.random.default_rng()

    inr_by_id = {}
    for item in usable:
        inr_by_id[item.object_id] = interference_to_noise_db(item, params)

    positive = [value for value in inr_by_id.values() if value > 0.0]
    max_inr_db = max(positive) if positive else 0.0
    total_inr_db = 10.0 * math.log10(
        sum(10.0 ** (max(value, -200.0) / 10.0) for value in inr_by_id.values())
    )

    desensitisation_db = 0.0
    if max_inr_db > 0.0:
        # Only the part of the interference above our own noise floor costs us
        # real target SNR, plus a fixed term for threshold drag and loading.
        desensitisation_db = max_inr_db + float(params.desensitisation_db)

    multiplier = (
        false_alarm_multiplier(max_inr_db, params) if max_inr_db > 0.0 else 1.0
    )

    unambiguous_range_m = params.unambiguous_range_m
    phantoms = []
    for item in usable:
        inr_db = inr_by_id[item.object_id]
        if inr_db <= 0.0:
            continue
        probability = phantom_probability(inr_db, params)
        if probability <= 0.0 or rng.random() >= probability:
            continue

        # Order 0 is the interferer itself; order k is the range-ambiguity
        # replica at |k * R_unamb - d|, which for an unambiguous range around
        # 150 m paints a phantom well beyond the real target.
        for order in range(int(params.max_ghost_orders) + 1):
            if order == 0:
                apparent_range_m = item.distance_m
            else:
                apparent_range_m = abs(order * unambiguous_range_m - item.distance_m)
            if apparent_range_m <= 0.0:
                continue
            phantoms.append(
                {
                    "parent_object_id": item.object_id,
                    "ghost_order": order,
                    "distance_m": apparent_range_m,
                    "azimuth_rad": item.azimuth_rad,
                    "relative_velocity_mps": item.relative_velocity_mps,
                    "inr_db": inr_db,
                    # Later ambiguity orders are weaker: the comb is not flat.
                    "snr_db": inr_db - 3.0 * order,
                    "label": item.label,
                }
            )

    return RRIReport(
        max_inr_db=max_inr_db,
        total_inr_db=total_inr_db,
        desensitisation_db=desensitisation_db,
        false_alarm_multiplier=multiplier,
        phantom_count=len(phantoms),
        phantoms=tuple(phantoms),
        interferer_count=len(usable),
    )
