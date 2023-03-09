import torch
import torch.nn
import segmentation_models_pytorch as smp
import numpy as np
import fsspec


def load_weights(path, map_location=None):
    fs = fsspec.filesystem("gs")
    with fs.open(path, "rb") as f:
        weights = torch.load(f, map_location=map_location)

    return weights


def find_padding(v, divisor=8):
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def padded_predict(tensor:np.array, model:torch.nn.Module, divisor:int=32,
                   device:torch.device=torch.device("cpu")) -> np.array:

    pad_r = find_padding(tensor.shape[-2], divisor)
    pad_c = find_padding(tensor.shape[-1], divisor)

    tensor_padded = np.pad(
        tensor, ((0, 0), (pad_r[0], pad_r[1]), (pad_c[0], pad_c[1])), "reflect"
    )

    slice_rows = slice(pad_r[0], None if pad_r[1] <= 0 else -pad_r[1])
    slice_cols = slice(pad_c[0], None if pad_c[1] <= 0 else -pad_c[1])

    tensor_padded = torch.tensor(tensor_padded, device=device)[None]  # Add batch dim

    with torch.no_grad():
        pred_padded = model(tensor_padded)[0]
        if len(pred_padded.shape) == 3:
            pred_cont = pred_padded[(slice(None), slice_rows, slice_cols)]
        elif len(pred_padded.shape) == 2:
            pred_cont = pred_padded[(slice_rows, slice_cols)]
        else:
            raise NotImplementedError(f"Don't know how to slice the tensor of shape {pred_padded.shape}")


    return np.array(pred_cont.cpu())


class CDModel(torch.nn.Module):
    """
    Example:
        model = CDModel(device=torch.device("cpu"))
        weights_file = "cloudSEN12model.ckpt"
        weights = load_weights(weights_file, map_location="cpu")
        model.load_state_dict(weights["state_dict"])
    """
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=13,
            classes=4
        )
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    def forward(self, tensor:torch.Tensor) -> torch.Tensor:
        pred_cont = self.model(tensor)
        pred_discrete = torch.argmax(pred_cont, dim=1).type(torch.uint8)
        return pred_discrete

    def predict(self, tensor: np.array) -> np.array:
        """
            tensor: np.array (13, H, W)

        Returns:
            long np.array (H, W) with interpretation {0: clear, 1: Thick cloud, 2: thin cloud, 3: cloud shadow}
        """
        assert tensor.shape[0] == 13, f"Expected 13 channels found {tensor.shape[0]}"

        return padded_predict(tensor, self, 32, self.device)

