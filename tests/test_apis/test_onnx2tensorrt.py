import os.path as osp
import tempfile

import mmcv
import pytest
import torch
import torch.nn as nn

from mmdeploy.apis.tensorrt import is_available

onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx').name
engine_file = tempfile.NamedTemporaryFile(suffix='.engine').name
test_img = torch.rand([1, 3, 8, 8])

trt_skip = not is_available()
cuda_skip = not torch.cuda.is_available()


@pytest.mark.skip(reason='This a not test class but a utility class.')
class TestModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5


test_model = TestModel().eval().cuda()


def get_deploy_cfg():
    import tensorrt as trt
    deploy_cfg = mmcv.Config(
        dict(
            backend_config=dict(
                type='tensorrt',
                common_config=dict(
                    fp16_mode=False,
                    log_level=trt.Logger.INFO,
                    max_workspace_size=1 << 30),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            input=dict(
                                min_shape=[1, 3, 8, 8],
                                opt_shape=[1, 3, 8, 8],
                                max_shape=[1, 3, 8, 8])))
                ])))
    return deploy_cfg


def generate_onnx_file(model):
    with torch.no_grad():
        dynamic_axes = {
            'input': {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'output': {
                0: 'batch'
            }
        }
        torch.onnx.export(
            model,
            test_img,
            onnx_file,
            output_names=['output'],
            input_names=['input'],
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=False,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        assert osp.exists(onnx_file)


@pytest.mark.skipif(trt_skip, reason='TensorRT not avaiable')
@pytest.mark.skipif(cuda_skip, reason='Cuda not avaiable')
def test_onnx2tensorrt():
    from mmdeploy.apis.tensorrt import load_trt_engine, onnx2tensorrt
    model = test_model
    generate_onnx_file(model)
    deploy_cfg = get_deploy_cfg()

    work_dir, save_file = osp.split(engine_file)
    onnx2tensorrt(work_dir, save_file, 0, deploy_cfg, onnx_file)
    assert osp.exists(work_dir)
    assert osp.exists(engine_file)
    engine = load_trt_engine(engine_file)
    assert engine is not None