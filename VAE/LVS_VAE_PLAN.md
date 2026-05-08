# Latent Value Shaping Variational Autoencoder (LVS-VAE)
# VAE-Based Value Shaping for PPO in Tigers and Goats

## 📌 Project Goal

Train a Variational Autoencoder (VAE) with a value head to learn a **latent representation of board states** and use it to **improve PPO training via reward shaping**.

---

## 🧠 Core Idea

* Learn a function:

  ```
  state → latent representation → value
  ```

* Then use that learned value to guide PPO:

  ```
  r_total = r_env + α [V(s') - V(s)]
  ```

---

## 🔄 Full Pipeline

### 1. Data Collection

* Run games using multiple agents:

  * Random policies
  * Early PPO checkpoints
  * Mid-training models
  * Strong trained models
  * Different tiger opponents (greedy, smart, trained)

* Save per state:

  ```
  state
  winner
  move_index
  total_moves
  progress_ratio
  goats_eaten
  goats_placed
  phase
  ```

---

### 2. Dataset Construction

* Each move = one training sample

  ```
  s0, s1, ..., sT
  ```

* Assign labels based on outcome:

  ```
  goat win  → 1.0
  tiger win → 0.0
  tie       → 0.5
  ```

* Target dataset size:

  ```
  20k (demo)
  40k–60k (final)
  ```

* Balance dataset:

  ```
  ~40% goat wins
  ~40% tiger wins
  ~20% ties
  ```

---

### 3. Endgame Filtering (Optional)

Define:

```
progress_ratio = move_index / total_moves
```

Endgame condition:

```
progress_ratio >= 0.7
```

Or include:

```
goats_eaten >= 4
tiger_moves <= 3
```

---

## 🧩 VAE Architecture

### Encoder

```
state → μ(s), logσ²(s)
```

### Sampling

```
z = μ + σϵ,   ϵ ~ N(0, I)
```

### Decoder

```
z → reconstructed state ŝ
```

### Value Head

```
μ(s) → V(s)
```

---

## 📉 Loss Function

```
L = L_recon + β L_KL + λ L_value
```

Where:

* Reconstruction loss: state reconstruction
* KL loss: regularizes latent space
* Value loss: supervised outcome prediction

### Hyperparameters

```
β = 0.001 – 0.01
λ = 1.0
latent_dim = 8 or 16
```

---

## 🧠 Latent Space Interpretation

* Each state is encoded as:

  ```
  q(z|s) = N(μ(s), diag(σ²(s)))
  ```

* Latent space represents:

  ```
  compressed strategic structure
  ```

* Value is extracted from latent:

  ```
  V(s) = f(μ(s))
  ```

---

## 📊 Value Function

Represents:

```
V(s) ≈ P(goat wins | state s)
```

Optional centering:

```
V_centered = 2V - 1
```

---

## 🧊 Frozen VAE Integration

### After training:

* Freeze encoder + value head

```
V(s) = ValueHead(Encoder(s).μ)
```

### PPO Reward Shaping

```
r_total = r_env + α [V(s') - V(s)]
```

Where:

```
α = 0.05 – 0.2
```

---

## ⚙️ PPO Interaction

* PPO still trains:

  * Actor (policy)
  * Critic (value function)

* VAE acts as:

```
an auxiliary reward signal
```

* Effects:

  * Improves credit assignment
  * Provides dense reward
  * Encourages better state transitions

---

## 📈 Experiments

Compare:

```
Baseline PPO
vs
PPO + VAE reward shaping
```

Metrics:

```
goat win rate
tiger win rate
timeouts
episode length
training stability
convergence speed
```

---

## 🎯 Deliverables

### Required

```
VAE + value head trained offline
Evaluation of value prediction
```

### Final Goal

```
Frozen VAE used to assist PPO training
```

---

## 🚀 Optional Extensions

* Endgame-specific VAE
* Latent space visualization (PCA / t-SNE)
* Confidence weighting using reconstruction error
* Sequence/chain modeling

---

## 🧭 Key Insights

* Latent space ≠ value
* Latent space = strategic representation
* Value = function of latent
* Reconstruction ensures meaningful encoding
* PPO uses value differences, not absolute value

---

## ✅ Final Summary

This project builds a **learned evaluator** for Tigers and Goats using a VAE and integrates it into PPO as a **reward shaping signal**, improving learning efficiency and strategic understanding.
