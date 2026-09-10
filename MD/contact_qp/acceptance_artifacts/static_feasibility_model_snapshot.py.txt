"""Independent, unilateral finite-area test plant; never a controller model.

Vertical spring elements, rigid TCP rotation, Kelvin–Voigt damping, nonuniform
stiffness, Coulomb friction, exogenous surface motion and delayed actuation.
Only the observation renderer knows acoustic disturbances. Synthetic geometry
and signal response are deliberately explicit; no patient inference is made.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .features import FeatureConfig

SCENARIOS = ("healthy", "left_gap", "right_gap", "both_edges", "central_gap", "shadow",
             "wrong_direction", "nonmonotonic", "curvature", "moving_surface",
             "delayed_execution", "hard_contact")


@dataclass(frozen=True)
class PlantConfig:
    schema_version: int = 1
    dt_s: float = .005
    half_length_m: float = .025
    elements: int = 81
    stiffness_n_m: float = 800.
    damping_n_s_m: float = 8.
    friction: float = .18
    rocking_friction_nm: float = .004
    actuator_tau_s: float = .012
    actuator_delay_s: float = .025
    path_length_m: float = .06
    scan_speed_m_s: float = .02
    warmup_s: float = 1.5
    image_period_s: float = .04
    image_delay_s: float = .152
    feature_processing_delay_s: float = .006
    image_width: int = 32
    image_height: int = 48
    coupling_depth_m: float = .00012
    force_noise_n: float = .015
    torque_noise_nm: float = .00015


class FiniteAreaPlant:
    def __init__(self, scenario, seed, config=None):
        if scenario not in SCENARIOS:
            raise ValueError("unknown independent scenario")
        self.config = config or PlantConfig()
        self.scenario, self.seed = scenario, int(seed)
        cfg = self.config
        rng = np.random.default_rng(np.random.SeedSequence([SCENARIOS.index(scenario), seed]))
        self.x = np.linspace(-cfg.half_length_m, cfg.half_length_m, cfg.elements)
        self.u = self.x / cfg.half_length_m
        weights = np.exp(.18*rng.standard_normal(cfg.elements))
        total_k = cfg.stiffness_n_m * (4. if scenario == "hard_contact" else 1.)
        self.k = weights / weights.sum() * total_k
        self.d = self.k / total_k * cfg.damping_n_s_m
        self.phase = rng.uniform(0., 2*np.pi)
        self.texture = np.clip(92.+15.*rng.standard_normal((cfg.image_height, cfg.image_width)), 10, 170)
        self.t = 0.
        self.path = 0.
        self.world_x = 0.
        self.z = 4./total_k
        self.theta = 0.
        self.velocity = np.zeros(6)
        self.held = np.zeros(6)
        self.commands = deque()
        self.pressures = np.zeros(cfg.elements)
        self.depth = np.zeros(cfg.elements)
        self.wrench_environment = np.zeros(6)
        self._update_contact()

    @property
    def scan_time(self):
        return max(0., self.t-self.config.warmup_s)

    def surface(self, path=None, time=None, world_x=None):
        s = self.path if path is None else float(path)
        t = self.scan_time if time is None else float(time)
        points_x = self.world_x+self.x*math.cos(self.theta) if world_x is None else np.asarray(world_x)
        u = points_x/self.config.half_length_m
        phase = np.clip((s-.007)/.046, 0., 1.)
        bump = math.sin(np.pi*phase)**2
        left = .5*(1.-np.tanh((u+.22)/.2))
        right = .5*(1.+np.tanh((u-.22)/.2))
        name = self.scenario
        h = np.zeros_like(u)
        if name in ("left_gap", "wrong_direction", "nonmonotonic", "delayed_execution", "hard_contact"):
            h += .008*bump*left
        elif name == "right_gap":
            h += .008*bump*right
        elif name == "both_edges":
            h += .008*bump*u*u
        elif name == "central_gap":
            h += .010*bump*np.exp(-(u/.20)**2)
        elif name == "curvature":
            h += bump*(.004*u*u + .005*math.sin(2*np.pi*s/.06)*u)
        elif name == "moving_surface":
            h += bump*(.0035*u*math.sin(2*np.pi*.7*t+self.phase)
                       + .0006*math.sin(2*np.pi*1.1*t+self.phase))
        return h

    def _update_contact(self):
        cfg = self.config
        c = math.cos(self.theta)
        height = self.surface()
        eps = 1e-5
        points_x = self.world_x+self.x*c
        surface_slope = (self.surface(self.path+eps)-self.surface(self.path-eps))/(2*eps)
        surface_x_slope = (self.surface(world_x=points_x+eps)-self.surface(world_x=points_x-eps))/(2*eps)
        surface_time_rate = (self.surface(time=self.scan_time+eps)-self.surface(time=self.scan_time-eps))/(2*eps)
        sx = math.sin(self.theta)
        world_vx = c*self.velocity[0]+sx*self.velocity[2]
        world_vz = -sx*self.velocity[0]+c*self.velocity[2]
        surface_rate = (surface_slope*self.velocity[1]+surface_time_rate
                        + surface_x_slope*(world_vx-self.x*sx*self.velocity[4]))
        self.depth = self.z-self.x*math.sin(self.theta)-height
        speed = world_vz-self.x*c*self.velocity[4]-surface_rate
        self.pressures = np.where(self.depth > 0., np.maximum(0., self.k*self.depth+self.d*speed), 0.)
        vertical_force = float(self.pressures.sum())
        torque = float(np.dot(self.x*(c-sx*surface_x_slope), self.pressures))
        friction_y = -cfg.friction*vertical_force*math.tanh(self.velocity[1]/.002)
        spin_friction = -cfg.rocking_friction_nm*math.tanh(self.velocity[4]/.01)
        # A static sloping surface exchanges spring work through scanning too.
        # Prescribed surface motion is a separate external plant energy input.
        reaction_y = float(self.pressures @ surface_slope)
        reaction_x = float(self.pressures @ surface_x_slope)
        self.surface_power_w = -float(self.pressures @ surface_time_rate)
        # Environment-on-tool physical port, expressed at TCP in tool axes.
        self.wrench_environment = np.array([c*reaction_x+sx*vertical_force, reaction_y+friction_y,
                                           sx*reaction_x-vertical_force*c, 0., torque+spin_friction, 0.])

    def control_wrench(self, *, noise=True):
        out = -self.wrench_environment.copy()
        if noise:
            out[2] += self.config.force_noise_n*math.sin(2*np.pi*13.1*self.t+self.phase)
            out[4] += self.config.torque_noise_nm*math.sin(2*np.pi*8.3*self.t+self.phase)
        return out

    def pose(self):
        return np.array([self.world_x, self.path, self.z, 0., self.theta, 0.])

    def command(self, twist):
        v = np.asarray(twist, dtype=float).reshape(6).copy()
        if not np.isfinite(v).all():
            raise ValueError("invalid plant command")
        if abs(v[3])+abs(v[5]) > 1e-14:
            raise ValueError("reduced plant supports TCP translations and y rocking only")
        delay = .060 if self.scenario == "delayed_execution" else self.config.actuator_delay_s
        self.commands.append((self.t+delay, v))

    def step(self):
        cfg = self.config
        self.t += cfg.dt_s
        while self.commands and self.commands[0][0] <= self.t+1e-12:
            _, self.held = self.commands.popleft()
        self.velocity += (1-math.exp(-cfg.dt_s/cfg.actuator_tau_s))*(self.held-self.velocity)
        self.path += cfg.dt_s*self.velocity[1]
        c, sx = math.cos(self.theta), math.sin(self.theta)
        self.world_x += cfg.dt_s*(c*self.velocity[0]+sx*self.velocity[2])
        self.z += cfg.dt_s*(-sx*self.velocity[0]+c*self.velocity[2])
        self.theta += cfg.dt_s*self.velocity[4]
        self._update_contact()

    def window_coupling(self, feature_config=None):
        cfg = feature_config or FeatureConfig()
        image_fraction = .5*(self.u+1)
        coupled = self.depth >= self.config.coupling_depth_m
        return np.array([coupled[(image_fraction >= lo) & (image_fraction < hi)].mean()
                         for lo, hi in cfg.lateral_windows])

    def _acoustically_available(self, rays, depth):
        available = depth >= self.config.coupling_depth_m
        if self.scenario == "shadow":
            available &= rays >= -.004
        if self.scenario == "nonmonotonic":
            available &= depth <= .008
        return available

    def window_acoustic_coupling(self, feature_config=None):
        """Independent latent acoustic truth, including persistent shadows.

        Retained separately from the frozen mechanical-coupling metric.
        It controls gap/outcome labels; confidence scores never define truth.
        """
        cfg = feature_config or FeatureConfig()
        available = self._acoustically_available(self.x, self.depth)
        fraction = .5*(self.u+1)
        return np.array([available[(fraction >= lo) & (fraction < hi)].mean()
                         for lo, hi in cfg.lateral_windows])

    def render(self, frame_index):
        """The controller receives pixels only; acoustic stressors are hidden."""
        cfg = self.config
        rays = np.linspace(-cfg.half_length_m, cfg.half_length_m, cfg.image_width)
        depth = np.interp(rays, self.x, self.depth)
        disconnected = ~self._acoustically_available(rays, depth)
        im = np.roll(self.texture, int(frame_index) % cfg.image_height, axis=0).copy()
        im *= np.exp(-.3*np.linspace(0, 1, cfg.image_height))[:, None]
        im[2:, disconnected] *= .012
        im[:2, disconnected] = 215.
        if self.scenario == "wrong_direction":
            im = im[:, ::-1]
        return np.asarray(np.clip(im, 0, 255), dtype=np.uint8)
