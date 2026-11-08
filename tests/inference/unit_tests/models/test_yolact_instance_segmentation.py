import numpy as np
import pytest

from inference.models.yolact.yolact_instance_segmentation import YOLACT


@pytest.mark.parametrize("img_dim", [(550, 550), (480, 640), (640, 480)])
def test_decode_masks_returns_masks_shaped_like_the_network_input(
    img_dim: tuple,
) -> None:
    # given
    np.random.seed(0)
    proto = np.random.rand(69, 69, 32).astype(np.float32)
    masks = np.random.rand(2, 32).astype(np.float32)
    boxes = np.array(
        [[0, 0, img_dim[1], img_dim[0]], [0, 0, img_dim[1] // 2, img_dim[0] // 2]],
        dtype=np.float32,
    )

    # when
    result = YOLACT.decode_masks(None, boxes, masks, proto, img_dim)

    # then
    assert result.shape == (2, img_dim[0], img_dim[1])
