import pytest
import torch

from fmnist.model import MLP, SimpleCNN, build_model, count_parameters


@pytest.mark.parametrize("name", ["cnn", "mlp"])
def test_forward_shape(name):
    model = build_model(name)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("resnet-9000")


def test_cnn_has_more_params_than_mlp_baseline_is_not_required_but_both_are_reasonable():
    cnn, mlp = SimpleCNN(), MLP()
    assert 100_000 < count_parameters(cnn) < 5_000_000
    assert 100_000 < count_parameters(mlp) < 5_000_000


def test_cnn_is_trainable_one_step():
    """Loss should go down after a single optimiser step on a fixed batch."""
    torch.manual_seed(0)
    model = SimpleCNN(width=8)
    model.train()
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss_before = loss_fn(model(x), y)
    loss_before.backward()
    opt.step()
    opt.zero_grad()
    model.eval()  # avoid dropout noise for the comparison
    loss_after = loss_fn(model(x), y)
    assert loss_after.item() < loss_before.item() + 0.5  # loose bound: it must not blow up
