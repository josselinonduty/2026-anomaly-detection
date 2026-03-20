from anomalib.data import Visa
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Patchcore

N = 50

datamodule = Visa(
    root="./datasets/visa",
    category="candle",
    train_batch_size=16,
    eval_batch_size=16,
)

_original_setup = datamodule.setup

def _setup_with_limit(stage=None):
    _original_setup(stage)
    for attr in ("train_data", "val_data", "test_data"):
        split = getattr(datamodule, attr, None)
        if split is not None:
            n = min(N, len(split))
            split.subsample(list(range(n)), inplace=True)


datamodule.setup = _setup_with_limit

model = Patchcore(
    num_neighbours=6,
)

engine = Engine(
    max_epochs=10,
)

engine.fit(datamodule=datamodule, model=model)

test_results = engine.test(datamodule=datamodule, model=model)

engine.export(
    model=model,
    export_type=ExportType.ONNX,
)
engine.export(
    model=model,
    export_type=ExportType.OPENVINO,
)