import tempfile
from pathlib import Path

import pytest

from inference_model_manager.hash_namespacing import namespace_client_hash_id
from inference_server import configuration
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.legacy.test_action_recognition import clip_base64, write_clip

MODEL_ID = "sam3trackervideo"
API_KEY = "k"
RLE = {"size": [2, 2], "counts": "04"}
PROMPTS = [
    {"frame_idx": 0, "obj_id": 1, "x": 10.0, "y": 12.0, "label": 1},
    {"frame_idx": 3, "obj_id": 2, "x": 30.0, "y": 8.0, "label": 1},
    {"frame_idx": 3, "obj_id": 2, "x": 31.0, "y": 9.0, "label": 0},
]


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    return scratch


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4")


def _gateway(fake_stat, model_class_name="SAM3TrackerVideo"):
    fake_stat[MODEL_ID] = ("instance-segmentation", "embed_video")

    def _embed(image, params):
        frame = params["frames"][0]
        return {
            "video_id": params["video_id"],
            "n_frames": len(params["frames"]),
            "height": frame.shape[0],
            "width": frame.shape[1],
        }

    def _segment(image, params):
        return {"frame_idx": params["frame_idx"], "masks": {"1": RLE}}

    def _propagate(image, params):
        return [
            {"frame_idx": i, "masks": {"1": RLE, "2": RLE}}
            for i in params["frame_indices"] or range(3)
        ]

    gateway = FakeGateway(
        predictions={
            (MODEL_ID, "embed_video"): _embed,
            (MODEL_ID, "segment_video_frame"): _segment,
            (MODEL_ID, "propagate_video"): _propagate,
        },
        model_info={
            MODEL_ID: {
                "model_class_name": model_class_name,
                "actions": {
                    "embed_video": {},
                    "segment_video_frame": {},
                    "propagate_video": {},
                },
            }
        },
    )

    return gateway


def _infer_calls(gateway):
    return [call for call in gateway.calls if call[0] == "infer"]


def test_embed_decodes_every_frame_and_scopes_the_id_to_the_key(
    legacy_client, fake_stat, clip
):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/sam3/video/embed",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "video": {"type": "base64", "value": clip_base64(clip)},
            "encode_frames": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["video_id"] == "clip-1"
    assert body["n_frames"] == 25
    assert (body["height"], body["width"]) == (48, 64)
    [call] = _infer_calls(gateway)
    assert call[1:3] == (MODEL_ID, "embed_video")
    params = call[3]
    assert params["video_id"] == namespace_client_hash_id("clip-1", API_KEY)
    assert params["encode_frames"] is True
    assert len(params["frames"]) == 25
    assert params["frames"][0].shape == (48, 64, 3)


def test_embed_refuses_a_clip_over_the_frame_cap(
    legacy_client, fake_stat, clip, monkeypatch
):
    monkeypatch.setattr(configuration, "SAM3_VIDEO_MAX_FRAMES", 10)
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/sam3/video/embed",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "video": {"type": "base64", "value": clip_base64(clip)},
        },
    )

    assert response.status_code == 413, response.text
    assert _infer_calls(gateway) == []


def test_segment_passes_every_click_and_the_frame(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/sam3/video/segment",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "prompts": PROMPTS,
            "frame_idx": 3,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frame_idx"] == 3
    assert body["masks"] == {"1": RLE}
    [call] = _infer_calls(gateway)
    assert call[2] == "segment_video_frame"
    assert call[3] == {
        "video_id": namespace_client_hash_id("clip-1", API_KEY),
        "prompts": PROMPTS,
        "frame_idx": 3,
    }


def test_propagate_returns_the_asked_frames(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/sam3/video/propagate",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "prompts": PROMPTS,
            "frame_indices": [0, 2, 4],
        },
    )

    assert response.status_code == 200, response.text
    frames = response.json()["frames"]
    assert [f["frame_idx"] for f in frames] == [0, 2, 4]
    assert frames[0]["masks"] == {"1": RLE, "2": RLE}
    [call] = _infer_calls(gateway)
    assert call[2] == "propagate_video"
    assert call[3]["frame_indices"] == [0, 2, 4]


def test_a_click_label_other_than_0_or_1_is_refused(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)
    prompts = [dict(PROMPTS[0], label=2)]

    response = legacy_client(gateway).post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": prompts},
    )

    assert response.status_code == 422, response.text
    assert _infer_calls(gateway) == []


def test_routes_refuse_a_model_that_is_not_the_video_tracker(legacy_client, fake_stat):
    gateway = _gateway(fake_stat, model_class_name="SAM3Torch")

    response = legacy_client(gateway).post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 400, response.text
    assert "SAM3TrackerVideo" in response.json()["message"]
    assert _infer_calls(gateway) == []
