import numpy as np
import pytest

pytest.importorskip("ultralytics")

from uav_detection.detection_node import (  # noqa: E402
    intersect_ray_with_horizontal_plane,
    project_bbox_footpoint_to_horizontal_plane,
)


def test_horizontal_plane_intersection_rejects_upward_ray() -> None:
    origin = np.array([0.0, 0.0, 4.0])
    direction = np.array([1.0, 0.0, 0.1])

    assert (
        intersect_ray_with_horizontal_plane(
            origin,
            direction,
            0.0,
            min_distance_m=0.5,
            max_distance_m=50.0,
        )
        is None
    )


def test_bbox_footpoint_projects_to_ground_plane() -> None:
    camera_origin = np.array([0.0, 0.0, 4.0])
    # Camera optical z points world +x, optical x points world +y,
    # optical y points world -z.
    camera_rotation_world = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ])

    result = project_bbox_footpoint_to_horizontal_plane(
        x1=40.0,
        y1=40.0,
        x2=60.0,
        y2=70.0,
        image_width=100,
        image_height=100,
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        camera_origin_world=camera_origin,
        camera_rotation_world=camera_rotation_world,
        plane_z=0.0,
        min_distance_m=0.5,
        max_distance_m=50.0,
        bbox_ground_y_fraction=1.0,
    )

    assert result is not None
    point, distance = result
    assert point[0] == pytest.approx(20.0)
    assert point[1] == pytest.approx(0.0)
    assert point[2] == pytest.approx(0.0)
    assert distance == pytest.approx(np.linalg.norm(point - camera_origin))
