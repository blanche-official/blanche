import datetime as dt
from dataclasses import fields

from blanche.actions.base import Action, SpecialAction
from blanche.actions.space import ActionSpace, SpaceCategory
from blanche.browser.observation import Observation
from blanche.browser.snapshot import SnapshotMetadata
from blanche.data.space import DataSpace, ImageData
from blanche.sdk.types import ActionSpaceResponse, ObserveResponse, SessionResponse


def test_observation_fields_match_response_types():
    """
    Ensure all fields in Observation have corresponding fields in ObserveResponseDict/ObserveResponse.
    This test will fail if a new field is added to Observation but not to the response types.
    """
    # Get all field names from Observation
    observation_fields = {field.name for field in fields(Observation)}

    # Remove internal fields that start with '_'
    observation_fields = {f for f in observation_fields if not f.startswith("_")}

    # Create a sample observation with all fields filled
    sample_data = {
        "metadata": {
            "url": "https://example.com",
            "title": "Test Page",
            "timestamp": dt.datetime.now(),
        },
        "screenshot": b"fake_screenshot",
        "data": {
            "markdown": "test data",
        },
    }

    # Try to create ObserveResponseDict with these fields
    response_dict = {
        "session": {
            "session_id": "test_session",  # Required by ResponseDict
            "timeout_minutes": 100,
            "created_at": dt.datetime.now(),
            "last_accessed_at": dt.datetime.now(),
            "duration": dt.timedelta(seconds=100),
            "status": "active",
        },
        **sample_data,
        "space": {
            "description": "test space",
            "actions": [],
            "category": None,
            "special_actions": SpecialAction.list(),
        },
    }

    # This will raise a type error if any required fields are missing
    response = ObserveResponse.model_validate(response_dict)

    # Check that all Observation fields exist in ObserveResponse
    response_fields = set(response.model_fields.keys())
    missing_fields = observation_fields - response_fields

    assert not missing_fields, f"Fields {missing_fields} exist in Observation but not in ObserveResponse"


def test_action_space_fields_match_response_types():
    """
    Ensure all fields in ActionSpace have corresponding fields in ActionSpaceResponseDict/ActionSpaceResponse.
    This test will fail if a new field is added to ActionSpace but not to the response types.
    """
    # Get all field names from ActionSpace
    space_fields = {field.name for field in fields(ActionSpace)}

    # Remove internal fields that start with '_' and known exclusions
    excluded_fields = {"_embeddings", "_actions"}  # _actions is 'actions' in the response types
    space_fields = {f for f in space_fields if not f.startswith("_") and f not in excluded_fields}
    space_fields.add("actions")  # Add back 'actions' without underscore

    # Create a sample space with all fields filled
    sample_data = {
        "description": "test space",
        "actions": [],
        "category": "homepage",
        "special_actions": SpecialAction.list(),
    }

    # Try to create ActionSpaceResponseDict with these fields
    response_dict = sample_data

    # This will raise a type error if any required fields are missing
    response = ActionSpaceResponse.model_validate(response_dict)

    # Check that all ActionSpace fields exist in ActionSpaceResponse
    response_fields = set(response.model_fields.keys())
    missing_fields = space_fields - response_fields

    assert not missing_fields, f"Fields {missing_fields} exist in ActionSpace but not in ActionSpaceResponse"


def test_observe_response_from_observation():
    obs = Observation(
        metadata=SnapshotMetadata(
            url="https://www.google.com",
            title="Google",
            timestamp=dt.datetime.now(),
        ),
        screenshot=b"fake_screenshot",
        data=DataSpace(
            markdown="test data",
            images=[
                ImageData(id="F1", url="https://www.google.com/image1.jpg"),
                ImageData(id="F2", url="https://www.google.com/image2.jpg"),
            ],
            structured=[{"key": "value"}],
        ),
        _space=ActionSpace(
            description="test space",
            category=SpaceCategory.OTHER,
            _actions=[
                Action(
                    id="L0",
                    description="my_test_description_0",
                    category="my_test_category_0",
                ),
                Action(
                    id="L1",
                    description="my_test_description_1",
                    category="my_test_category_1",
                ),
            ],
        ),
    )
    dt_now = dt.datetime.now()
    session = SessionResponse(
        session_id="test_session",
        timeout_minutes=100,
        created_at=dt_now,
        last_accessed_at=dt_now,
        duration=dt.timedelta(seconds=100),
        status="active",
    )

    response = ObserveResponse.from_obs(
        session=session,
        obs=obs,
    )
    assert response.session.session_id == "test_session"
    assert response.session.timeout_minutes == 100
    assert response.session.created_at == dt_now
    assert response.session.last_accessed_at == dt_now
    assert response.session.duration == dt.timedelta(seconds=100)
    assert response.session.status == "active"
    assert response.metadata.title == "Google"
    assert response.metadata.url == "https://www.google.com"
    assert response.screenshot == b"fake_screenshot"
    assert response.data is not None
    assert response.data.markdown == "test data"
    assert response.space is not None
    assert response.space.description == "test space"
    assert response.space.category == "other"
    assert response.space.actions == obs.space.actions()
