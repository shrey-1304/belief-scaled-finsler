# Belief-Scaled Finsler Geodesic Encoder for Seq2Seq Models

This repository implements a novel **Finsler geodesic encoder** as an alternative to traditional attention mechanisms for sequence-to-sequence models. The encoder uses differential geometry to learn context representations that can outperform standard attention-based encoders.

## Overview

### Core Concept

The project compares two encoder architectures for the MultiWOZ 2.2 dialogue dataset:

1. **Geodesic Encoder** (Novel): Integrates velocities along derived Finsler geodesics on a learned manifold
2. **Attention Encoder** (Baseline): Standard transformer attention with causal self-attention

Both encoders:
- Feed the same shared decoder
- Share a tied embedding table for fair comparison
- Produce identical memory slot dimensions
- Enable isolated encoder performance comparison

### Mathematical Foundation

#### Finsler Manifold Construction

The geodesic encoder builds a belief-scaled Finsler manifold:

```
t(x) = 0.9 * tanh(MLP(x))           # Belief field on the manifold
M(x) = M_base · Cayley(t(x) A)      # Deforms the local indicatrix
a(x) = M^(-T) M^(-1)                # Metric tensor field (defines the manifold)
F(x, v) = √(a·v·v) + c·v            # Randers norm (c ≠ 0 ⟹ irreversibility)
```

#### Geodesic Dynamics

The encoder solves the geodesic differential equation:

```
x' = v
v' = -2G(x, v)    # where G is the Randers spray with derived Christoffels
```

Where:
- **Christoffels** (γ): Derived from the metric tensor field
- **Randers spray** (G): Closed-form computation combining:
  - Christoffel symbols
  - Symmetric component (r): metric deformation effect
  - Antisymmetric component (s): irreversibility from Randers tilt

## Architecture Details

### Geodesic Encoder

**Learned Parameters:**
- Velocity generator: MLP that produces initial velocities from embeddings
- Belief MLP: Determines belief field t(x)
- Initial points x₀: Starting positions on the manifold
- Token embedding (tied with decoder)

**Fixed/Derived Components:**
- Asymmetric matrix A: Fixed randomly initialized matrix in GL(n, ℝ)
- Metric tensor a(x): Derived from belief field and matrix A
- Christoffel symbols: Analytically derived via finite differences
- Connection and spray: Derived from Christoffels using Randers geometry

**Integration:**
- Absorbs STRIDE tokens per geodesic step
- Accumulates velocities over strided token windows
- Emits (x, v) state at each integration step as a memory slot
- Output: [B, NSLOT, DMEM]

### Attention Encoder (Baseline)

**Architecture:**
1. Token embedding projection
2. RoPE (Rotary Position Embedding) positional encoding
3. Multi-head self-attention (4 heads)
4. MLP feedforward layer
5. Strided mean-pooling to match geodesic encoder's slot count
6. Projection to DMEM dimensions

**Design:** Matched capacity with geodesic encoder; both produce [B, NSLOT, DMEM]

### Shared Decoder

**Architecture:**
- Transformer decoder with 2 layers
- 4 attention heads
- Cross-attention to encoder memory
- Tied weight embedding with encoder
- Causal masking for autoregressive generation

## Configuration

```python
# Model dimensions
CTX_LEN  = 64      # Context tokens
RSP_LEN  = 64      # Response tokens
D        = 64      # Model width (embeddings, decoder)

# Geodesic construction
NV       = 8       # Manifold dimension n
MVEL     = 4       # Number of velocities (parallel geodesics)
STRIDE   = 8       # Tokens per geodesic integration step
DTAU     = 0.3     # Integration step size
VCLAMP   = 3.0     # Velocity norm ceiling (numerical stability)
TILT     = 0.4     # Randers tilt magnitude ||c|| (0 ⟹ Riemannian)
DMEM     = 64      # Memory slot width (2 * MVEL * NV)
NSLOT    = 8       # Memory slots (CTX_LEN // STRIDE)

# Decoder
DEC_LAYERS = 2     # Decoder transformer layers
DEC_HEADS  = 4     # Attention heads
```

## Usage

### Installation

```bash
# Required packages
pip install -q kaggle torch transformers

# For Google Colab, GPU acceleration available
```

### Dataset Setup

```python
# Download MultiWOZ 2.2 from Kaggle
!kaggle datasets download -d taejinwoo/multiwoz-22
!unzip -o multiwoz-22.zip

# Load dialogues
train_dir = "/content/MultiWOZ_2.2/train"
dialogues = []
for file in sorted(os.listdir(train_dir)):
    if file.endswith(".json"):
        with open(os.path.join(train_dir, file), "r") as f:
            dialogues.extend(json.load(f))
```

### Training

```python
# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2Seq(
    encoder_kind="geodesic",  # or "attention"
    seed=0,
    vocab_size=tokenizer.vocab_size,
    pad_id=tokenizer.pad_token_id
).to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        ctx_ids = batch["ctx_ids"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        rsp_ids = batch["rsp_ids"].to(device)
        
        logits, loss = model(ctx_ids, ctx_mask, rsp_ids)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss = {total_loss/len(train_loader):.4f}")

# Save model
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "tokenizer_name": "gpt2",
}, "geodesic_seq2seq.pt")
```

### Evaluation

```python
# Load trained model
checkpoint = torch.load("geodesic_seq2seq.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Compute validation loss
total_loss = 0
with torch.no_grad():
    for batch in val_loader:
        ctx_ids = batch["ctx_ids"].to(device)
        ctx_mask = batch["ctx_mask"].to(device)
        rsp_ids = batch["rsp_ids"].to(device)
        
        _, loss = model(ctx_ids, ctx_mask, rsp_ids)
        total_loss += loss.item()

avg_loss = total_loss / len(val_loader)
perplexity = math.exp(avg_loss)
print(f"Val Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
```

## Key Features

### Belief-Scaled Geometry

The manifold adapts based on learned belief field t(x):
- High belief: Strong metric deformation
- Low belief: Closer to Riemannian geometry
- Enables context-dependent representation learning

### Derived Christoffels

Unlike many geometric approaches, Christoffels are:
- **Analytically derived** from the metric tensor
- **Computed via finite differences** through the belief MLP
- **Numerically stable** with proper clamping and conditioning

### Randers Structure

The Randers Finsler norm captures:
- **Reversibility** (symmetric metric): Standard geometric distance
- **Irreversibility** (tilt vector c): Directional asymmetry in the space
- When `TILT = 0`, reduces to Riemannian geometry

### Numerical Stability

- Velocity norm clamping (VCLAMP)
- Metric tensor regularization
- Proper masking of padding tokens
- Careful numerical differentiation

## Files

- `finsler_writeup.pdf`: Detailed mathematical formulation and theoretical background
- `finslergpt.ipynb`: Google Colab notebook with complete implementation and training pipeline
- Data handling, preprocessing, and MultiWOZ 2.2 dataset utilities
- Tokenizer support (GPT-2 or word-level fallback)

## Experimental Setup

**Dataset:** MultiWOZ 2.2
- Task: Dialogue response generation
- Training pairs: Context/response from multi-turn conversations
- Context: Previous dialogue turns
- Response: System utterance to generate

**Comparison:** 
- Geodesic encoder vs. Attention encoder
- Same decoder, same embedding, same training
- Fair capacity matching

**Metrics:**
- Cross-entropy loss
- Perplexity (PPL)
- Validation on held-out dev set

## Mathematical References

### Finsler Geometry
- Randers spaces: Deformed Riemannian metrics with directional component
- Derived connection: Computed from metric derivatives
- Spray: Generalization of geodesic spray

### Implementation Details
- Cayley transform: For matrix deformation in SO(n)
- Christoffel computation: Finite difference approximation through belief MLP
- Riemannian vs. Randers: Controlled via TILT parameter

## Future Work

- Multi-velocity integration with alternative schemes
- Learned metric tensor optimization
- Adaptive integration step sizing
- Application to other sequence modeling tasks
- Comparison with other geometric deep learning approaches

## Requirements

- Python 3.7+
- PyTorch 1.9+
- transformers (for GPT-2 tokenizer)
- Kaggle API credentials (for dataset)
- CUDA (recommended for GPU acceleration)

## Author

Shreyas (shrey-1304)

## License

See repository for license information.

---

**Note:** This is an experimental implementation combining differential geometry with deep learning. The geodesic encoder demonstrates that geometric priors can be effective for sequence modeling, offering an alternative to purely attention-based architectures.
