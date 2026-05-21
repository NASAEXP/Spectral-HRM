import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
from pathlib import Path
from models.lm_head import LMHead, TiedFourierVocab, LearnedTokenFourierVocab


def _load_experiment_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "experiments" / "Experiment 15 - Checkpoint Weight Memory Tradeoff" / "checkpoint_weight_tradeoff.py"
    spec = importlib.util.spec_from_file_location("checkpoint_weight_tradeoff", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_experiment_15_locks_checkpoint_weight_for_survival_runs():
    experiment = _load_experiment_module()

    assert experiment.LOCKED_CHECKPOINT_WEIGHT is True
    assert experiment.LOCKED_CONTROL_CHECKPOINT_WEIGHT is False

class IdentityModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head_hint = {
            "in": {"dim": hidden_size, "init_std": 0.25},
            "out": {"dim": hidden_size, "init_std": 0.25},
        }
        self.create_cache = lambda **_kwargs: None
        self.compute_train_extra_args = lambda _state: {}

    def forward(self, carry, x, **_seq_info):
        return carry, x

def test_checkpoint_weight_gradients_and_outputs_match_exactly():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    vocab_size = 120
    hidden_size = 64
    vocab_modes = 16
    hidden_modes = 8
    
    # 1. Test TiedFourierVocab
    # Create reference model (checkpoint OFF)
    ref_vocab = TiedFourierVocab(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        init_std=0.25,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        checkpoint_weight=False,
    )
    
    # Create test model (checkpoint ON) with identical initial parameters
    test_vocab = TiedFourierVocab(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        init_std=0.25,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        checkpoint_weight=True,
    )
    with torch.no_grad():
        test_vocab.coefficients.copy_(ref_vocab.coefficients)
        if ref_vocab.bias is not None:
            test_vocab.bias.copy_(ref_vocab.bias)
            
    # Dummy inputs
    inputs = torch.randint(0, vocab_size, (8,))
    hidden_states = torch.randn(8, hidden_size, requires_grad=True)
    hidden_states_ref = hidden_states.clone().detach().requires_grad_(True)
    
    # Forward Pass - Embedding
    ref_embed = ref_vocab.embed(inputs)
    test_embed = test_vocab.embed(inputs)
    
    assert torch.allclose(ref_embed, test_embed, atol=1e-6, rtol=1e-6), "Embeddings do not match"
    
    # Forward Pass - Logits (Training Mode)
    ref_vocab.train()
    test_vocab.train()
    
    ref_logits = ref_vocab.logits(hidden_states_ref)
    test_logits = test_vocab.logits(hidden_states)
    
    assert torch.allclose(ref_logits, test_logits, atol=1e-6, rtol=1e-6), "Logits do not match"
    
    # Backward Pass - check gradients
    loss_ref = ref_logits.sum()
    loss_test = test_logits.sum()
    
    loss_ref.backward()
    loss_test.backward()
    
    assert torch.allclose(ref_vocab.coefficients.grad, test_vocab.coefficients.grad, atol=1e-5, rtol=1e-5), "Gradients on coefficients do not match"
    assert torch.allclose(hidden_states_ref.grad, hidden_states.grad, atol=1e-5, rtol=1e-5), "Gradients on hidden states do not match"


def test_learned_token_fourier_checkpoint_gradients_match():
    torch.manual_seed(42)
    vocab_size = 100
    hidden_size = 32
    vocab_modes = 12
    hidden_modes = 6
    
    ref_vocab = LearnedTokenFourierVocab(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        init_std=0.25,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        checkpoint_weight=False,
    )
    
    test_vocab = LearnedTokenFourierVocab(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        init_std=0.25,
        vocab_modes=vocab_modes,
        hidden_modes=hidden_modes,
        checkpoint_weight=True,
    )
    with torch.no_grad():
        test_vocab.coefficients.copy_(ref_vocab.coefficients)
        test_vocab.token_basis.copy_(ref_vocab.token_basis)
        
    hidden_states = torch.randn(4, hidden_size, requires_grad=True)
    hidden_states_ref = hidden_states.clone().detach().requires_grad_(True)
    
    ref_vocab.train()
    test_vocab.train()
    
    ref_logits = ref_vocab.logits(hidden_states_ref)
    test_logits = test_vocab.logits(hidden_states)
    
    assert torch.allclose(ref_logits, test_logits, atol=1e-6, rtol=1e-6)
    
    ref_logits.sum().backward()
    test_logits.sum().backward()
    
    assert torch.allclose(ref_vocab.coefficients.grad, test_vocab.coefficients.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(ref_vocab.token_basis.grad, test_vocab.token_basis.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(hidden_states_ref.grad, hidden_states.grad, atol=1e-5, rtol=1e-5)


def test_lm_head_integration_with_checkpoint():
    torch.manual_seed(42)
    vocab_size = 50
    hidden_size = 16
    
    model = LMHead(
        IdentityModel(hidden_size=hidden_size),
        {
            "vocab_size": vocab_size,
            "vocab_head": {
                "type": "tied_fourier",
                "vocab_modes": 10,
                "hidden_modes": 8,
                "checkpoint_weight": True,
            },
        },
    )
    model.train()
    batch = {
        "inputs": torch.randint(0, vocab_size, (4,)),
        "labels": torch.randint(0, vocab_size, (4,)),
        "cu_seqlens": torch.tensor([0, 4]),
    }
    
    # Run forward/backward to make sure it doesn't crash and computes gradients
    carry, loss, metrics = model(carry=None, batch=batch)
    loss.backward()
    
    assert model.vocab_head.coefficients.grad is not None
