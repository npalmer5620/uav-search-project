"""Archimedean spiral waypoint generator for UAV search patterns."""

import math


class SpiralGenerator:
    """Generate Archimedean spiral waypoints in NED coordinates.

    The spiral equation is r = b * theta, where b = spacing / (2 * pi).
    Each call to step() advances theta and returns the next (x, y, z, yaw)
    setpoint. Pausing the spiral is as simple as not calling step() — the
    internal theta is preserved and resumes exactly where it left off.
    """

    def __init__(
        self,
        max_radius: float = 20.0,
        spacing: float = 5.0,
        angular_speed: float = 0.3,
        altitude: float = -10.0,
    ) -> None:
        self.max_radius = max_radius
        self.spacing = spacing
        self.angular_speed = angular_speed
        self.altitude = altitude
        self.b = spacing / (2.0 * math.pi)

        self.theta = 0.0
        self.complete = False

    def reset(self) -> None:
        """Restart the spiral from the beginning."""
        self.theta = 0.0
        self.complete = False

    def step(self, dt: float) -> tuple[float, float, float, float] | None:
        """Advance the spiral by *dt* seconds.

        Returns (x, y, z, yaw) in NED frame, or None if the spiral has
        exceeded max_radius.
        """
        if self.complete:
            return None

        self.theta += self.angular_speed * dt
        r = self.b * self.theta

        if r > self.max_radius:
            self.complete = True
            return None

        x = r * math.cos(self.theta)
        y = r * math.sin(self.theta)

        # Yaw from velocity direction
        dx = self.b * math.cos(self.theta) - r * math.sin(self.theta)
        dy = self.b * math.sin(self.theta) + r * math.cos(self.theta)
        yaw = math.atan2(dy, dx)

        return (x, y, self.altitude, yaw)

    @property
    def current_radius(self) -> float:
        return self.b * self.theta

    @property
    def progress(self) -> float:
        """Fraction complete (0.0 to 1.0) based on radius vs max_radius."""
        if self.max_radius <= 0.0:
            return 1.0
        return min(1.0, self.current_radius / self.max_radius)
