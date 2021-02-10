import itertools
import torch
import numpy as np


def padded_predict(predfunction, module_shape):
    def predict(x):
        shape_tensor = np.array(list(x.shape))[2:].astype(np.int64)
        shape_new_tensor = np.ceil(shape_tensor.astype(np.float32) / module_shape).astype(np.int64) * module_shape

        if np.all(shape_tensor == shape_new_tensor):
            return predfunction(x)

        pad_to_add = shape_new_tensor - shape_tensor
        refl_pad_layer = torch.nn.ReflectionPad2d((0, pad_to_add[1], 0, pad_to_add[0]))

        refl_pad_result = refl_pad_layer(x)
        pred_padded = predfunction(refl_pad_result)
        slice_ = (slice(None),
                  slice(None),
                  slice(0, shape_new_tensor[0]-pad_to_add[0]),
                  slice(0, shape_new_tensor[1]-pad_to_add[1]))

        return pred_padded[slice_]

    return predict


def predbytiles(pred_function, input_batch, tile_size=1280, pad_size=32, device=torch.device("cpu")):
    pred_continuous_tf = None
    assert input_batch.dim() == 4, "Expected batch of images"

    for b, i, j in itertools.product(range(0, input_batch.shape[0], tile_size),
                                     range(0, input_batch.shape[2], tile_size),
                                     range(0, input_batch.shape[3], tile_size)):

        slice_current = (slice(i, min(i + tile_size, input_batch.shape[2])),
                         slice(j, min(j + tile_size, input_batch.shape[3])))
        slice_pad = (slice(max(i - pad_size, 0), min(i + tile_size + pad_size, input_batch.shape[2])),
                     slice(max(j - pad_size, 0), min(j + tile_size + pad_size, input_batch.shape[3])))

        slice_save_i = slice(slice_current[0].start - slice_pad[0].start,
                             None if (slice_current[0].stop - slice_pad[0].stop) == 0 else slice_current[0].stop -
                                                                                           slice_pad[0].stop)
        slice_save_j = slice(slice_current[1].start - slice_pad[1].start,
                             None if (slice_current[1].stop - slice_pad[1].stop) == 0 else slice_current[1].stop -
                                                                                           slice_pad[1].stop)

        slice_save = (slice_save_i, slice_save_j)

        slice_prepend = (slice(b, b + 1), slice(None))
        slice_current = slice_prepend + slice_current
        slice_pad = slice_prepend + slice_pad
        slice_save = slice_prepend + slice_save

        vals_to_predict = input_batch[slice_pad]
        cnn_out = pred_function(vals_to_predict)

        assert cnn_out.dim() == 4, "Expected 4-band prediction (after softmax)"

        if pred_continuous_tf is None:
            pred_continuous_tf = torch.zeros((input_batch.shape[0], cnn_out.shape[1],
                                              input_batch.shape[2], input_batch.shape[3]),
                                             device=device)

        pred_continuous_tf[slice_current] = cnn_out[slice_save]

    return pred_continuous_tf
