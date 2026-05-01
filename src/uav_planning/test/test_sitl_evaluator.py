from uav_planning.sitl_evaluator import (
    ForwardFrustum,
    SearchGeometry,
    load_world_targets,
)


def test_load_world_people_and_victims_from_search_area():
    people = load_world_targets("worlds/search_area.sdf", target_filter="people")
    victims = load_world_targets("worlds/search_area.sdf", target_filter="victims")

    assert len(people) == 12
    assert len(victims) == 6
    assert all(target.class_name == "person" for target in people)
    assert all(target.is_victim for target in victims)


def test_search_bounds_filter_matches_default_grid():
    geometry = SearchGeometry(
        width_m=40.0,
        height_m=40.0,
        origin_north_m=0.0,
        origin_east_m=0.0,
        cell_size_m=4.0,
    )
    people = load_world_targets("worlds/search_area.sdf", target_filter="people")
    in_bounds = [
        target for target in people if geometry.contains_ned(target.ned_x, target.ned_y)
    ]

    assert len(in_bounds) == 6
    assert {target.name for target in in_bounds} == {
        "person_standing_01",
        "person_standing_02",
        "person_walking_01",
        "person_standing_03",
        "victim_standing_01",
        "victim_standing_02",
    }


def test_forward_frustum_respects_yaw_and_range():
    frustum = ForwardFrustum()

    assert frustum.is_visible_ned(
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_yaw_ned_rad=0.0,
        altitude_ned_m=-4.0,
        target_north_m=10.0,
        target_east_m=0.0,
    )
    assert not frustum.is_visible_ned(
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_yaw_ned_rad=0.0,
        altitude_ned_m=-4.0,
        target_north_m=0.0,
        target_east_m=10.0,
    )
    assert not frustum.is_visible_ned(
        drone_north_m=0.0,
        drone_east_m=0.0,
        drone_yaw_ned_rad=0.0,
        altitude_ned_m=-4.0,
        target_north_m=25.0,
        target_east_m=0.0,
    )
