"""SAM3 offline video tracking: embed a clip once, then click and propagate on it.

A clip is decoded frame by frame and kept by the SAM3 video tracker under the
caller's id, scoped to the caller's API key like SAM3 image embedding ids.
Segment and propagate calls name the clip by that id and carry every click on
it, so the tracker holds no per-object state between calls: only the clip
(its preprocessed frames and the encoder output of the frames seen so far).
That state lives in one worker, so the routes need the requests of one clip to
reach the same worker.
"""

import asyncio
import time
from typing import Any, List, Optional

import numpy as np

from inference_model_manager.hash_namespacing import namespace_client_hash_id
from inference_server import configuration
from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    Sam3VideoEmbedRequest,
    Sam3VideoEmbedResponse,
    Sam3VideoFrameMasks,
    Sam3VideoPropagateRequest,
    Sam3VideoPropagateResponse,
    Sam3VideoSegmentRequest,
    Sam3VideoSegmentResponse,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.video import read_frame_windows, video_source_path

SAM3_VIDEO_MODEL_CLASS = "SAM3TrackerVideo"


def ensure_sam3_video_route(model_id: str, route: Route) -> None:
    """Refuse a model that is not the SAM3 video tracker.

    Args:
        model_id: Model id as the caller named it, for the message.
        route: Resolved route of the model.

    Raises:
        LegacyHTTPError: 400 naming the class the model loads as instead.
    """
    if route.model_class_name == SAM3_VIDEO_MODEL_CLASS:
        return

    raise LegacyHTTPError(
        400,
        f"Model {model_id!r} loads as {route.model_class_name}; the SAM3 video "
        f"routes need a {SAM3_VIDEO_MODEL_CLASS} model.",
    )


def read_clip_frames(path: str, *, max_frames: int) -> List[np.ndarray]:
    """Decode every frame of a clip, in order.

    Args:
        path: File holding the clip.
        max_frames: Most frames a clip may have.

    Returns:
        The RGB ``HWC`` ``uint8`` frames.

    Raises:
        LegacyHTTPError: 400 when the clip has no frames, 413 when it has
            more than ``max_frames``.
    """
    # One window one frame longer than the cap: decoding stops there, so an
    # oversized clip is caught without decoding all of it.
    frames = next(read_frame_windows(path, windows=[range(max_frames + 1)]), [])
    if not frames:
        raise LegacyHTTPError(400, "The video has no frames.")
    if len(frames) > max_frames:
        raise LegacyHTTPError(
            413, f"The video has more than {max_frames} frames, the most allowed."
        )

    return frames


async def embed_video(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoEmbedRequest,
) -> Sam3VideoEmbedResponse:
    """Decode the request's clip and keep it with the tracker under its id.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The embed request.

    Returns:
        The clip's id, frame count and frame size.

    Raises:
        LegacyHTTPError: 400 for a clip that cannot be fetched or decoded, 413
            for one over the size or frame cap.
    """
    started = time.perf_counter()
    video = inference_request.video
    async with video_source_path(video.type, video.value) as path:
        frames = await asyncio.to_thread(
            read_clip_frames, path, max_frames=configuration.SAM3_VIDEO_MAX_FRAMES
        )

    described = await bridge.infer_params_only(
        route,
        api_key,
        "embed_video",
        {
            "frames": frames,
            "video_id": namespace_client_hash_id(inference_request.video_id, api_key),
            "encode_frames": inference_request.encode_frames,
        },
    )
    response = Sam3VideoEmbedResponse(
        video_id=inference_request.video_id,
        n_frames=described["n_frames"],
        height=described["height"],
        width=described["width"],
        time=time.perf_counter() - started,
    )

    return response


async def segment_video_frame(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoSegmentRequest,
) -> Sam3VideoSegmentResponse:
    """Masks on one frame of an embedded clip for the objects clicked there.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The segment request.

    Returns:
        The frame's masks by object id.
    """
    started = time.perf_counter()
    result = await bridge.infer_params_only(
        route,
        api_key,
        "segment_video_frame",
        {
            "video_id": namespace_client_hash_id(inference_request.video_id, api_key),
            "prompts": [prompt.model_dump() for prompt in inference_request.prompts],
            "frame_idx": inference_request.frame_idx,
        },
    )
    response = Sam3VideoSegmentResponse(
        frame_idx=result["frame_idx"],
        masks=result["masks"],
        time=time.perf_counter() - started,
    )

    return response


async def propagate_video(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoPropagateRequest,
) -> Sam3VideoPropagateResponse:
    """Track every clicked object over an embedded clip, forward and backward.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The propagate request.

    Returns:
        The masks of every returned frame, in frame order.
    """
    started = time.perf_counter()
    results = await bridge.infer_params_only(
        route,
        api_key,
        "propagate_video",
        {
            "video_id": namespace_client_hash_id(inference_request.video_id, api_key),
            "prompts": [prompt.model_dump() for prompt in inference_request.prompts],
            "frame_indices": inference_request.frame_indices,
        },
    )
    response = Sam3VideoPropagateResponse(
        frames=[Sam3VideoFrameMasks(**frame) for frame in results],
        time=time.perf_counter() - started,
    )

    return response
