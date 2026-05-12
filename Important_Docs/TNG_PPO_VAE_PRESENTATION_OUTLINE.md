# Tigers and Goats PPO + VAE Presentation Outline

## Slide 1: Project Goal

**Title:** Learning Goat Strategy in Tigers and Goats

- Train a reinforcement learning goat agent for Tigers and Goats.
- Use MaskablePPO so the policy only selects legal actions.
- Explore whether learned VAE reward shaping can reduce reliance on hand-built rewards.

**Speaker note:** The main question is whether a learned state representation can guide early and late training better than only manual reward terms.

## Slide 2: Tigers and Goats

**Title:** Game Setup

- Two-player asymmetric board game.
- Tigers try to capture goats.
- Goats try to trap or block tiger movement.
- The goat player needs strong opening placement and long-term survival structure.

**Speaker note:** The goat problem is difficult because many early moves only matter much later.

## Slide 3: PPO Training Setup

**Title:** Reinforcement Learning Baseline

- PPO learns from repeated self-play or scripted-opponent games.
- Action masking removes illegal actions before sampling.
- Reward includes sparse outcomes plus shaping terms.
- Metrics tracked: win rate, episode length, reward components, and value estimates.

**Speaker note:** PPO can learn from sparse wins and losses, but training is easier when the reward gives useful intermediate feedback.

## Slide 4: Reward Shaping Problem

**Title:** Why Add Learned Shaping?

- Manual shaping rewards help early training.
- Examples: center control, spacing, blocking, avoiding goat captures.
- But hand-built rewards can overfit to one opponent or strategy.
- Goal: learn a shaping signal from data instead of encoding every pattern by hand.

**Speaker note:** The VAE is used as a learned evaluator, not as the policy itself.

## Slide 5: Variational Autoencoder

**Title:** VAE Concept

- Encoder compresses a board state into a small latent vector.
- Decoder reconstructs the original state.
- KL loss keeps the latent space smooth.
- Extra prediction heads can learn useful labels from the latent state.

**Speaker note:** Reconstruction forces the latent space to preserve board structure, while heads attach task meaning to that structure.

## Slide 6: LVS-VAE Value Shaping

**Title:** Outcome Value Head

- Input: 25 state features.
- Latent size: 8 dimensions.
- Value head predicts goat-favorable outcome value.
- PPO reward adds a small bonus for positive value delta:

```text
vae_reward = coefficient * (value_after - value_before)
```

**Speaker note:** This makes the VAE a potential-style shaping signal: moves are rewarded when they improve predicted goat favorability.

## Slide 7: Phase-Gated Value Ensemble

**Title:** Full-Game and Endgame VAEs

- Full-game VAE handles general game structure.
- Endgame VAE specializes in late-game states.
- A progress gate blends in endgame value after a threshold.
- This avoids asking one model to explain every phase equally well.

**Speaker note:** The endgame signal may not activate often in short games, so tracking episode length and gate activity matters.

## Slide 8: Placing-Survival VAE

**Title:** Opening-Phase Survival Head

- Separate `LVS_VAE_PS` model focuses only on placing-phase survival.
- Trained on placing-phase states.
- Ties are blended with wins against losing states.
- Intended to learn early structures that prevent collapse.

**Speaker note:** This model is separated from the value VAE so the opening signal does not need endgame-only data.

## Slide 9: Experiments

**Title:** Ablation Plan

- Baseline PPO without VAE shaping.
- PPO with LVS-VAE value shaping.
- PPO with grouped manual rewards removed.
- PPO with placing-survival shaping enabled.
- Compare fast ballpark results before deeper tuning.

**Speaker note:** The goal is not just higher training win rate; it is whether the agent becomes more robust across opponents.

## Slide 10: Metrics

**Title:** What to Measure

- Goat win rate and tiger win rate.
- Episode length.
- Sparse reward vs VAE reward contribution.
- `lvs_value_delta` and `reward_vae_component`.
- Placing-survival delta when opening shaping is enabled.
- Generalization against multiple tiger opponents.

**Speaker note:** The VAE reward should be a useful hint, not the dominant objective.

## Slide 11: Current Direction

**Title:** Design Decisions

- Keep `LVS_VAE` as value-shaping only.
- Use separate `LVS_VAE_PS` for placing survival.
- Train value and survival models separately for simpler analysis.
- Use future state-action VAE heads for move-specific opening quality.

**Speaker note:** Separating the VAEs makes the deadline workflow simpler and keeps each model's purpose clearer.

## Slide 12: Conclusion

**Title:** Main Contribution

- PPO learns the goat policy with legal-action masking.
- VAEs provide learned reward shaping from board-state structure.
- Value VAE guides general outcome improvement.
- Placing-survival VAE targets early-game structure.
- The project tests whether learned latent shaping can replace some handcrafted rewards.

**Speaker note:** The long-term goal is a more general goat agent that is less dependent on opponent-specific manual reward engineering.
