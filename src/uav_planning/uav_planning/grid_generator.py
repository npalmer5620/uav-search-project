"""Boustrophedon (lawnmower) grid waypoint generator for UAV search patterns."""

import math


class GridGenerator:
    """Generate lawnmower-pattern waypoints over a rectangular area in NED.

    The pattern consists of parallel north-south legs spaced ``spacing``
    metres apart in the east direction.  Odd legs fly northbound, even legs
    fly southbound, producing a boustrophedon (serpentine) sweep.

    Each call to :meth:`step` advances the current position along the path
    at the configured *speed* and returns the next ``(x, y, z, yaw)``
    setpoint.  Pausing is as simple as not calling ``step()`` — the
    internal state is preserved and resumes exactly where it left off.
    """

    def __init__(
        self,
        width: float = 40.0,
        height: float = 40.0,
        spacing: float = 5.0,
        speed: float = 2.0,
        altitude: float = -4.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.width = width
        self.height = height
        self.spacing = spacing
        self.speed = speed
        self.altitude = altitude
        self.origin = origin

        self._waypoints: list[tuple[float, float]] = []
        self._segment_lengths: list[float] = []
        self._segment_yaws: list[float] = []
        self._total_length: float = 0.0

        self._build_waypoints()

        self._seg_idx: int = 0
        self._seg_t: float = 0.0  # distance travelled within current segment
        self.complete: bool = False

    # ------------------------------------------------------------------
    # Waypoint construction
    # ------------------------------------------------------------------

    def _build_waypoints(self) -> None:
        ox, oy = self.origin
        x_min = ox - self.height / 2.0
        x_max = ox + self.height / 2.0
        y_min = oy - self.width / 2.0
        y_max = oy + self.width / 2.0

        pts: list[tuple[float, float]] = []
        y = y_min
        leg = 0
        while y <= y_max + 1e-9:
            if leg % 2 == 0:
                pts.append((x_min, y))
                pts.append((x_max, y))
            else:
                pts.append((x_max, y))
                pts.append((x_min, y))
            leg += 1
            y += self.spacing

        # Collapse consecutive duplicates that arise at leg junctions
        self._waypoints = [pts[0]]
        for pt in pts[1:]:
            if math.dist(pt, self._waypoints[-1]) > 1e-6:
                self._waypoints.append(pt)

        # Pre-compute segment metadata
        self._segment_lengths.clear()
        self._segment_yaws.clear()
        self._total_length = 0.0
        for i in range(len(self._waypoints) - 1):
            ax, ay = self._waypoints[i]
            bx, by = self._waypoints[i + 1]
            length = math.hypot(bx - ax, by - ay)
            yaw = math.atan2(by - ay, bx - ax)
            self._segment_lengths.append(length)
            self._segment_yaws.append(yaw)
            self._total_length += length

    # ------------------------------------------------------------------
    # Public API (mirrors SpiralGenerator)
    # ------------------------------------------------------------------

    @property
    def current_position(self) -> tuple[float, float, float, float] | None:
        """Return the current target ``(x, y, z, yaw)`` without advancing."""
        if self.complete or not self._segment_lengths:
            return None

        frac = self._seg_t / self._segment_lengths[self._seg_idx]
        ax, ay = self._waypoints[self._seg_idx]
        bx, by = self._waypoints[self._seg_idx + 1]
        x = ax + frac * (bx - ax)
        y = ay + frac * (by - ay)
        yaw = self._segment_yaws[self._seg_idx]
        return (x, y, self.altitude, yaw)

    def step(self, dt: float) -> tuple[float, float, float, float] | None:
        """Advance by *dt* seconds. Returns ``(x, y, z, yaw)`` or ``None``."""
        if self.complete:
            return None

        if not self._segment_lengths:
            self.complete = True
            return None

        advance = self.speed * dt
        self._seg_t += advance

        # Walk through segments until we've consumed the advance distance
        while self._seg_t >= self._segment_lengths[self._seg_idx]:
            self._seg_t -= self._segment_lengths[self._seg_idx]
            self._seg_idx += 1
            if self._seg_idx >= len(self._segment_lengths):
                self.complete = True
                return None

        return self.current_position

    def reset(self) -> None:
        """Restart the grid from the beginning."""
        self._seg_idx = 0
        self._seg_t = 0.0
        self.complete = False

    @property
    def progress(self) -> float:
        """Fraction complete (0.0 to 1.0) based on distance travelled."""
        if self._total_length <= 0.0:
            return 1.0
        travelled = sum(self._segment_lengths[:self._seg_idx]) + self._seg_t
        return min(1.0, travelled / self._total_length)
