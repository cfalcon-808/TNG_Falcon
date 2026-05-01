# Network Architecture and Model Combiner Concept

This note explains the current `MaskablePPO` network layout used by `model_summary_abc.py` / `train_abc.py`, then sketches a rough way to combine two trained models by learning a small neural network on top of their outputs.

## Current Model Shape

The active model is a `sb3_contrib.MaskablePPO` policy using an MLP actor-critic architecture.

At a high level:

```text
TnGEnv observation
  |
  v
policy.features_extractor
  |
  v
feature vector
  |
  +-------------------------------+
  |                               |
  v                               v
actor / policy branch             critic / value branch
policy.mlp_extractor.policy_net   policy.mlp_extractor.value_net
  |                               |
  v                               v
policy.action_net                 policy.value_net
  |                               |
  v                               v
action logits                     scalar value estimate
```

## Current Default Layer Sizes

The current training script builds fresh models with:

```text
policy_kwargs = dict(net_arch=[256, 256, 256])
```

The environment defines:

```text
observation_space = Box(..., shape=(25,))
action_space = Discrete(115)
```

So the default architecture is approximately:

```text
Input observation: 25 values

Shared:
  policy.features_extractor
    input:  25
    output: 25

Actor branch:
  policy.mlp_extractor.policy_net
    Linear(25 -> 256)
    Tanh
    Linear(256 -> 256)
    Tanh
    Linear(256 -> 256)
    Tanh

  policy.action_net
    Linear(256 -> 115)

  actor output:
    115 action logits

Critic branch:
  policy.mlp_extractor.value_net
    Linear(25 -> 256)
    Tanh
    Linear(256 -> 256)
    Tanh
    Linear(256 -> 256)
    Tanh

  policy.value_net
    Linear(256 -> 1)

  critic output:
    1 scalar value estimate
```

In table form:

| Branch | Module | Layer | Input size | Output size | Meaning |
| --- | --- | --- | --- | --- | --- |
| Shared | `policy.features_extractor` | flatten / feature pass-through | 25 | 25 | Converts env observation into policy features |
| Actor | `policy.mlp_extractor.policy_net` | `Linear` | 25 | 256 | First actor hidden layer |
| Actor | `policy.mlp_extractor.policy_net` | `Tanh` | 256 | 256 | Actor nonlinearity |
| Actor | `policy.mlp_extractor.policy_net` | `Linear` | 256 | 256 | Second actor hidden layer |
| Actor | `policy.mlp_extractor.policy_net` | `Tanh` | 256 | 256 | Actor nonlinearity |
| Actor | `policy.mlp_extractor.policy_net` | `Linear` | 256 | 256 | Third actor hidden layer |
| Actor | `policy.mlp_extractor.policy_net` | `Tanh` | 256 | 256 | Actor nonlinearity |
| Actor | `policy.action_net` | `Linear` | 256 | 115 | Final actor head, one logit per action |
| Critic | `policy.mlp_extractor.value_net` | `Linear` | 25 | 256 | First critic hidden layer |
| Critic | `policy.mlp_extractor.value_net` | `Tanh` | 256 | 256 | Critic nonlinearity |
| Critic | `policy.mlp_extractor.value_net` | `Linear` | 256 | 256 | Second critic hidden layer |
| Critic | `policy.mlp_extractor.value_net` | `Tanh` | 256 | 256 | Critic nonlinearity |
| Critic | `policy.mlp_extractor.value_net` | `Linear` | 256 | 256 | Third critic hidden layer |
| Critic | `policy.mlp_extractor.value_net` | `Tanh` | 256 | 256 | Critic nonlinearity |
| Critic | `policy.value_net` | `Linear` | 256 | 1 | Final critic head, state value `V(s)` |

Important distinction:

```text
policy.action_net output size = 115
```

This is the actor. It outputs one logit for each possible encoded action.

```text
policy.value_net output size = 1
```

This is the critic. It outputs one value estimate for the current state.

Saved models may differ if they were trained with a different `net_arch`, custom policy settings, or older code. For a saved model, trust the output of:

```powershell
python model_summary_abc.py --model-path path\to\model.zip --architecture-only
```

## Mapping Torchsummary Rows to Named Layers

When `torchsummary` prints the full policy, it may show rows like this:

```text
Flatten-1                   [-1, 25]               0
FlattenExtractor-2          [-1, 25]               0
Linear-3                    [-1, 256]          6,656
Tanh-4                      [-1, 256]              0
Linear-5                    [-1, 256]         65,792
Tanh-6                      [-1, 256]              0
Linear-7                    [-1, 256]         65,792
Tanh-8                      [-1, 256]              0
Linear-9                    [-1, 256]          6,656
Tanh-10                     [-1, 256]              0
Linear-11                   [-1, 256]         65,792
Tanh-12                     [-1, 256]              0
Linear-13                   [-1, 256]         65,792
Tanh-14                     [-1, 256]              0
MlpExtractor-15   [[-1, 256], [-1, 256]]           0
Linear-16                   [-1, 1]              257
Linear-17                   [-1, 115]         29,555
```

The named mapping is:

| Torchsummary row | Named module path | Branch | Meaning |
| --- | --- | --- | --- |
| `Flatten-1` | inside `policy.features_extractor` | Shared | Flattens the 25-value observation |
| `FlattenExtractor-2` | `policy.features_extractor` | Shared | SB3 feature extractor wrapper, outputs feature vector `[25]` |
| `Linear-3` | `policy.mlp_extractor.policy_net[0]` | Actor | Actor hidden layer 1: `25 -> 256` |
| `Tanh-4` | `policy.mlp_extractor.policy_net[1]` | Actor | Actor activation after hidden layer 1 |
| `Linear-5` | `policy.mlp_extractor.policy_net[2]` | Actor | Actor hidden layer 2: `256 -> 256` |
| `Tanh-6` | `policy.mlp_extractor.policy_net[3]` | Actor | Actor activation after hidden layer 2 |
| `Linear-7` | `policy.mlp_extractor.policy_net[4]` | Actor | Actor hidden layer 3: `256 -> 256` |
| `Tanh-8` | `policy.mlp_extractor.policy_net[5]` | Actor | Actor activation after hidden layer 3 |
| `Linear-9` | `policy.mlp_extractor.value_net[0]` | Critic | Critic hidden layer 1: `25 -> 256` |
| `Tanh-10` | `policy.mlp_extractor.value_net[1]` | Critic | Critic activation after hidden layer 1 |
| `Linear-11` | `policy.mlp_extractor.value_net[2]` | Critic | Critic hidden layer 2: `256 -> 256` |
| `Tanh-12` | `policy.mlp_extractor.value_net[3]` | Critic | Critic activation after hidden layer 2 |
| `Linear-13` | `policy.mlp_extractor.value_net[4]` | Critic | Critic hidden layer 3: `256 -> 256` |
| `Tanh-14` | `policy.mlp_extractor.value_net[5]` | Critic | Critic activation after hidden layer 3 |
| `MlpExtractor-15` | `policy.mlp_extractor` | Split point | Returns two vectors: actor latent `[256]` and critic latent `[256]` |
| `Linear-16` | `policy.value_net` | Critic head | Final critic output: `256 -> 1`, the state value `V(s)` |
| `Linear-17` | `policy.action_net` | Actor head | Final actor output: `256 -> 115`, one logit per action |

The same mapping as a diagram:

```text
Input observation[25]
  |
  v
Flatten-1 / FlattenExtractor-2
policy.features_extractor
  |
  v
feature vector[25]
  |
  +---------------------------------------------+
  |                                             |
  v                                             v
Actor branch                                   Critic branch
policy.mlp_extractor.policy_net               policy.mlp_extractor.value_net
  |                                             |
  +-- Linear-3   25 -> 256                      +-- Linear-9    25 -> 256
  +-- Tanh-4                                     +-- Tanh-10
  +-- Linear-5  256 -> 256                      +-- Linear-11  256 -> 256
  +-- Tanh-6                                     +-- Tanh-12
  +-- Linear-7  256 -> 256                      +-- Linear-13  256 -> 256
  +-- Tanh-8                                     +-- Tanh-14
  |                                             |
  v                                             v
actor latent[256]                              critic latent[256]
  |                                             |
  v                                             v
Linear-17                                      Linear-16
policy.action_net                             policy.value_net
256 -> 115                                    256 -> 1
  |                                             |
  v                                             v
115 action logits                             1 state-value estimate
```

One detail that can look odd in the `torchsummary` output: `Linear-16` appears before `Linear-17`. That is just the order SB3 calls the heads during the policy forward pass. It does not mean the critic comes before the actor conceptually; they are sibling heads after the actor/critic split.

## Observation Input

The environment observation is currently:

```text
board[23] + goats_eaten + phase
```

So the default observation shape is usually:

```text
25
```

Meaning:

- 23 board-position values
- 1 goats-eaten value
- 1 phase value

For an MLP policy, the features extractor is usually simple. It mostly flattens or forwards the observation into a feature vector that the actor and critic branches can consume.

## Feature-to-Diagram Mapping

The left side of the diagram starts with the 25-value observation vector:

```text
TnGEnv observation[25]
  |
  +-- board[0]
  +-- board[1]
  +-- board[2]
  +-- ...
  +-- board[22]
  +-- goats_eaten
  +-- phase
```

Those 25 values map into the diagram like this:

```text
board[0..22], goats_eaten, phase
  |
  v
policy.features_extractor
  |
  v
feature vector[25]
  |
  +-------------------------------+
  |                               |
  v                               v
actor / policy branch             critic / value branch
```

The individual input features are:

| Feature | Index / range | Size | Goes to | Meaning |
| --- | --- | --- | --- | --- |
| `board` | `0..22` | 23 | shared input, then both branches | Current piece layout on the 23 board points |
| `goats_eaten` | `23` | 1 | shared input, then both branches | Number of goats captured by tigers |
| `phase` | `24` | 1 | shared input, then both branches | Whether the game is in placing phase or moving phase |

The actor and critic both receive all 25 features. There are not separate actor-only or critic-only input features in the current default setup.

More explicitly:

```text
Input vector[25]
  |
  +-- board[0..22]
  |     tells both branches where pieces are
  |
  +-- goats_eaten
  |     tells both branches how close the tiger side is to the capture win condition
  |
  +-- phase
        tells both branches whether goat placement is still happening or normal movement has begun
```

Then the branches use the same input for different purposes:

| Feature group | Actor uses it to help decide | Critic uses it to help estimate |
| --- | --- | --- |
| `board[0..22]` | Which legal move or placement is best | Whether the current position is strong or weak |
| `goats_eaten` | Whether to prioritize survival, capture, blocking, or pressure | How close the state is to a tiger win or goat survival path |
| `phase` | Whether the action should be interpreted as placement or movement | Whether the state value should be judged as early placement or later movement |

In the full network diagram:

```text
Input features:
  board[0..22] + goats_eaten + phase
        |
        v
Shared feature extractor:
  policy.features_extractor
        |
        v
Shared feature vector[25]
        |
        +----------------------------------+
        |                                  |
        v                                  v
Actor branch                              Critic branch
policy_net                               value_net
        |                                  |
        v                                  v
action_net                               value_net head
        |                                  |
        v                                  v
115 action logits                         1 state-value estimate
```

## Shared Front End

### `policy.features_extractor`

This is the shared front end.

```text
observation -> features_extractor -> feature vector
```

Both the actor and critic receive the result of this step.

In this project's default MLP setup, this is not the main learned decision-making body. The important learned layers are mostly in the actor and critic MLP branches.

## Actor Path

The actor is the part that chooses actions.

```text
observation
  -> policy.features_extractor
  -> policy.mlp_extractor.policy_net
  -> policy.action_net
  -> action logits
  -> action mask / action distribution
  -> selected action
```

### `policy.mlp_extractor.policy_net`

This is the actor's hidden network.

With a net architecture like:

```text
256,256,256
```

the actor branch is conceptually:

```text
feature vector
  -> Linear(..., 256)
  -> activation
  -> Linear(256, 256)
  -> activation
  -> Linear(256, 256)
  -> activation
```

This branch learns patterns that are useful for deciding which move should be chosen.

### `policy.action_net`

This is the actor's final output head.

```text
actor hidden vector -> Linear(256, number_of_actions)
```

For this environment, the action space is currently:

```text
Discrete(115)
```

So the action head usually outputs:

```text
115 logits
```

Each logit is a raw score for one encoded action. These are not probabilities yet.

### Action Mask and Softmax

It is normal not to see a visible `Softmax` layer in the PyTorch module tree.

The model usually outputs raw action logits:

```text
policy.action_net -> logits
```

Then `MaskablePPO` handles the action distribution outside the visible `nn.Module` tree:

```text
logits -> apply invalid-action mask -> probability distribution -> sample/select action
```

So:

- `policy.action_net` is the actor output layer
- it outputs one score per possible action
- invalid actions are masked later
- softmax-like probability conversion happens in the distribution logic, not as a normal `nn.Softmax` layer

## Critic Path

The critic is the part that estimates how good the current state is.

```text
observation
  -> policy.features_extractor
  -> policy.mlp_extractor.value_net
  -> policy.value_net
  -> scalar value estimate V(s)
```

### `policy.mlp_extractor.value_net`

This is the critic's hidden network.

With a net architecture like:

```text
256,256,256
```

the critic branch is conceptually:

```text
feature vector
  -> Linear(..., 256)
  -> activation
  -> Linear(256, 256)
  -> activation
  -> Linear(256, 256)
  -> activation
```

This branch learns patterns that are useful for judging board states.

### `policy.value_net`

This is the critic's final output head.

```text
critic hidden vector -> Linear(256, 1)
```

The layer that outputs `1` is the critic head.

That one number is:

```text
V(s)
```

Meaning:

```text
estimated value of the current state
```

The actor answers:

```text
Which action should I take?
```

The critic answers:

```text
How good is this state?
```

## Actor vs Critic Summary

| Part | Path | Output | Meaning |
| --- | --- | --- | --- |
| Shared front end | `policy.features_extractor` | feature vector | Converts observation into features |
| Actor hidden layers | `policy.mlp_extractor.policy_net` | actor hidden vector | Learns action-selection features |
| Actor head | `policy.action_net` | one logit per action | Scores possible actions |
| Critic hidden layers | `policy.mlp_extractor.value_net` | critic hidden vector | Learns state-value features |
| Critic head | `policy.value_net` | one scalar | Estimates `V(s)` |

## Two-Model Combiner Idea

The rough idea is to take two trained models and learn a small combiner network on top of them.

Suppose there are two trained policies:

```text
Model A
Model B
```

Each model can produce:

```text
actor logits
critic value estimate
possibly hidden features
```

The simplest combiner would use the actor logits from both models:

```text
observation
  |
  +-> Model A -> logits_A
  |
  +-> Model B -> logits_B

concat(logits_A, logits_B)
  -> combiner network
  -> combined logits
  -> action mask
  -> final action distribution
```

For this environment:

```text
logits_A shape: 115
logits_B shape: 115
combined input shape: 230
combined output shape: 115
```

The combiner is learning:

```text
When should I trust Model A?
When should I trust Model B?
When should I blend both?
When should I override both with a new nonlinear combination?
```

## Combiner With Skip Connections

A useful rough design is to let the combiner produce a new set of logits, while also allowing direct skip paths from each original model.

Conceptually:

```text
logits_A -----------------------------+
                                      |
logits_B -----------------------------+
                                      |
concat(logits_A, logits_B)            |
  -> combiner hidden layers           |
  -> combiner_delta_logits            |
                                      |
final logits =                        |
  gate_A * logits_A                   |
  + gate_B * logits_B                 |
  + gate_C * combiner_delta_logits
```

This allows three behaviors:

- choose mostly Model A
- choose mostly Model B
- use a learned nonlinear correction on top of both

The gates could be:

```text
fixed weights
learned global weights
learned per-state weights
learned per-action weights
```

The fastest rough version would probably use learned per-state gates:

```text
concat(logits_A, logits_B, value_A, value_B, observation)
  -> gate network
  -> gate_A, gate_B, gate_C
```

Then:

```text
final logits = weighted combination of A, B, and combiner output
```

## Why Use Logits Instead of Actions

Combining final chosen actions is crude because each model only gives one decision.

Combining logits is richer because each model gives its full preference profile:

```text
Model A may strongly prefer action 10 but also like action 24.
Model B may strongly reject action 10 but prefer action 31.
```

The combiner can learn from those relative preferences.

Using logits also preserves uncertainty better than just using the final sampled action.

## Possible Combiner Inputs

A minimal combiner could use:

```text
logits_A
logits_B
```

A better combiner could use:

```text
logits_A
logits_B
value_A
value_B
observation
action_mask
```

A more advanced combiner could also use hidden features:

```text
actor_hidden_A
actor_hidden_B
critic_hidden_A
critic_hidden_B
```

That would give the combiner more internal information, but it is more coupled to the exact model architecture.

## Rough Combiner Architecture

One possible conceptual layout:

```text
Input:
  logits_A[115]
  logits_B[115]
  value_A[1]
  value_B[1]
  observation[25]
  action_mask[115]

Total rough input:
  115 + 115 + 1 + 1 + 25 + 115 = 372

Combiner:
  Linear(372, 256)
  activation
  Linear(256, 256)
  activation

Outputs:
  combined_logits[115]
  gate_A
  gate_B
  gate_combiner
```

Then:

```text
final_logits =
  gate_A * logits_A
  + gate_B * logits_B
  + gate_combiner * combined_logits
```

Finally:

```text
final_logits -> apply action mask -> action distribution
```

## Training Strategy Options

### Option 1: Freeze Both Models

The fastest version:

```text
Model A frozen
Model B frozen
Combiner trainable
```

Pros:

- simpler
- faster
- less risk of destroying either trained model
- easier to debug

Cons:

- combiner can only reuse what A and B already know
- cannot improve internal features of A or B

### Option 2: Freeze First, Then Fine-Tune

Practical staged approach:

```text
Stage 1: freeze A and B, train combiner only
Stage 2: optionally unfreeze some upper layers
Stage 3: train slowly with a small learning rate
```

Pros:

- starts stable
- allows later improvement

Cons:

- more moving parts
- higher risk of overfitting or degrading the stronger model

### Option 3: Distillation-Style Training

The combiner could be trained to imitate whichever model performs better in a given state.

Example target:

```text
if Model A wins this position more often, imitate A
if Model B wins this position more often, imitate B
otherwise blend them
```

This requires collecting comparison data.

## Important Cautions

### Critic Values Are Not Directly Comparable Unless Training Matches

`value_A` and `value_B` may not be on the same scale if the models were trained with different:

- reward settings
- learner roles
- opponents
- curricula
- discount factors
- normalization behavior

So the combiner should be careful when using critic values. Actor logits are often a cleaner first input.

### Logit Scales Can Differ

One model may output sharper logits than the other.

Example:

```text
Model A logits: small range, uncertain
Model B logits: large range, very confident
```

The combiner may need normalization or learned gates so one model does not dominate just because its logits have larger magnitude.

### Action Masks Must Still Be Applied

The final combined logits must still respect legal actions.

The combiner should not be allowed to choose invalid actions.

The usual order should be:

```text
combine logits first
apply action mask second
sample/select action third
```

### Combining Two Weak Models Does Not Guarantee a Strong Model

The combiner can only exploit useful differences between the models if those differences are visible in the inputs and represented in training data.

The best case is:

```text
Model A is good in some positions
Model B is good in different positions
Combiner learns when each one is better
```

The worst case is:

```text
Both models make the same mistake
Combiner learns the same mistake
```

## Recommended First Experiment

A rough but controlled first experiment:

1. Freeze both trained models.
2. Feed the combiner:
   - `logits_A`
   - `logits_B`
   - current observation
   - action mask
3. Output `combined_logits`.
4. Apply the normal action mask.
5. Train only the combiner.
6. Compare against:
   - Model A alone
   - Model B alone
   - simple average of A and B logits
   - max-confidence pick between A and B

The simple baselines matter. If a learned combiner cannot beat averaging or picking the more confident model, the combiner is probably not learning anything useful yet.

## Mental Model

Think of each original PPO model as an expert:

```text
Expert A says: here are my scores for every legal move.
Expert B says: here are my scores for every legal move.
```

The combiner is a referee:

```text
Given the board and both experts' score sheets, decide the final score sheet.
```

The final action still comes from the same kind of masked action distribution used by `MaskablePPO`.
