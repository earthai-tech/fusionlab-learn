# tests/test_infer_pinn_mode.py
import numpy as np
import pytest

from fusionlab.nn.pinn.op import infer_pinn_mode

# Fixtures – simple tensor-like placeholders (NumPy is enough here)
@pytest.fixture
def dummy_tensor():
    return np.zeros((1,))           # shape/content irrelevant


def test_infer_mode_dict(dummy_tensor):
    inputs = {
        "coords": dummy_tensor,
        "dynamic_features": dummy_tensor,
    }
    assert infer_pinn_mode(inputs) == "as_dict"


# List / tuple input  → "as_list"
@pytest.mark.parametrize("container_type", [list, tuple])
def test_infer_mode_sequence(container_type, dummy_tensor):
    inputs = container_type([dummy_tensor, dummy_tensor])
    assert infer_pinn_mode(inputs) == "as_list"


#Invalid input type  → TypeError
def test_infer_mode_invalid():
    with pytest.raises(TypeError):
        infer_pinn_mode(42)    # int is neither Mapping nor Sequence

if __name__=='__main__': 
    pytest.main ( [__file__]) 
    