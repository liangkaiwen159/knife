from typing import Dict, Iterable, Optional

import ncnn
import numpy as np
import torch

from mmdeploy.apis.ncnn import ncnn_ext
from mmdeploy.utils.timer import TimeCounter


class NCNNWrapper(torch.nn.Module):
    """NCNN wrapper class for inference.

    Args:
        param_file (str): Path of a parameter file.
        bin_file (str): Path of a binary file.
        output_names (list[str] | tuple[str]): Names to model outputs. Defaults
            to `None`.

    Examples:
        >>> from mmdeploy.apis.ncnn import NCNNWrapper
        >>> import torch
        >>>
        >>> param_file = 'model.params'
        >>> bin_file = 'model.bin'
        >>> model = NCNNWrapper(param_file, bin_file)
        >>> inputs = dict(input=torch.randn(1, 3, 224, 224))
        >>> outputs = model(inputs)
        >>> print(outputs)
    """

    def __init__(self,
                 param_file: str,
                 bin_file: str,
                 output_names: Optional[Iterable[str]] = None,
                 **kwargs):
        super(NCNNWrapper, self).__init__()

        net = ncnn.Net()
        ncnn_ext.register_mm_custom_layers(net)
        net.load_param(param_file)
        net.load_model(bin_file)

        self._net = net
        self._output_names = output_names

    def set_output_names(self, output_names: Iterable[str]):
        """Set names of the model outputs.

        Args:
            output_names (list[str] | tuple[str]): Names to model outputs.
        """
        self._output_names = output_names

    def get_output_names(self):
        """Get names of the model outputs.

        Returns:
            list[str]: Names to model outputs.
        """
        if self._output_names is not None:
            return self._output_names
        else:
            assert hasattr(self._net, 'output_names')
            return self._net.output_names()

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """Run forward inference.

        Args:
            inputs (Dict[str, torch.Tensor]): Key-value pairs of model inputs.

        Returns:
            Dict[str, torch.Tensor]: Key-value pairs of model outputs.
        """
        input_list = list(inputs.values())
        batch_size = input_list[0].size(0)
        for input_tensor in input_list[1:]:
            assert input_tensor.size(
                0) == batch_size, 'All tensors should have same batch size'
            assert input_tensor.device.type == 'cpu', \
                'NCNN only supports cpu device'

        # set output names
        output_names = self.get_output_names()

        # create output dict
        outputs = dict([name, [None] * batch_size] for name in output_names)

        # run inference
        for batch_id in range(batch_size):
            # create extractor
            ex = self._net.create_extractor()

            # set inputs
            for name, input_tensor in inputs.items():
                input_mat = ncnn.Mat(
                    input_tensor[batch_id].detach().cpu().numpy())
                ex.input(name, input_mat)

            # get outputs
            result = self.ncnn_execute(extractor=ex, output_names=output_names)
            for name in output_names:
                outputs[name][batch_id] = torch.from_numpy(
                    np.array(result[name]))

        # stack outputs together
        for name, input_tensor in outputs.items():
            outputs[name] = torch.stack(input_tensor)

        return outputs

    @TimeCounter.count_time()
    def ncnn_execute(self, extractor: ncnn.Extractor,
                     output_names: Iterable[str]):
        """Run inference with NCNN.

        Args:
            extractor (ncnn.Extractor): NCNN extractor to extract output.
            output_names (Iterable[str]): A list of string specifying
                output names.

        Returns:
            dict[str, ncnn.Mat]: Inference results of NCNN model.
        """
        result = {}
        for name in output_names:
            out_ret, out = extractor.extract(name)
            assert out_ret == 0, f'Failed to extract output : {out}.'
            result[name] = out
        return result