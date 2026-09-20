# Training Laya Vision to play games

Laya Vision can act as a game policy. Each step, the screen goes in as the image and a `choice` question whose options are the game's buttons picks the move. The question builders live in `laya/games.py`, shared by the live viewers and the training-data jobs so both ask exactly the same question.

- `examples/atari_live.py --game <Name>` plays any of the 104 Atari games in `ale-py` in a local window.
- `examples/vizdoom_live.py --scenario <name>` does the same for ViZDoom scenarios.

## What happens without game training

The released checkpoint was trained on photo and diagram questions, not games:

- **Breakout:** it never chose FIRE, so the ball never launched. The viewer now presses FIRE automatically at the start of each game and after each lost life, which is the standard Atari `FireResetEnv` trick. `--no-auto-fire` turns that off.
- **ViZDoom `basic`:** it chose ATTACK about 90% of the time wherever the monster was, and never strafed to line up the shot. It matches "monster visible → shoot" but doesn't relate the monster's position to the crosshair.

## Where training data can come from

### Best: generate it yourself with an expert

Run an expert in the emulator and save, for every frame, the full-colour screen and the expert's action. This beats public datasets because:

- The inputs match what the model sees live: same emulator settings, colours and resolution. Most public Atari data is 84×84 grayscale.
- An expert *policy* gives a probability over actions. Laya's loss (soft cross-entropy plus a proper scoring rule) accepts soft targets, so the model learns "RIGHT 70%, ATTACK 25%" and keeps meaningful confidence.
- You can make as much as you want.

Kinds of expert:

| Expert | Where to get one | Games |
|---|---|---|
| Scripted from game internals | ViZDoom's labels buffer gives every object's screen bounding box | Doom scenarios with a simple rule, such as `basic` (done: see below) |
| Pretrained RL agents | Hugging Face Hub: Stable-Baselines3 (`sb3/ppo-*`, `sb3/dqn-*`), CleanRL models, Sample Factory ViZDoom agents | Most Atari games; several Doom scenarios |
| RAM-based rules | Atari RAM holds object positions, e.g. the ball and paddle x in Breakout | Breakout, Pong, Freeway |

### Public datasets

| Dataset | Contents | Fit |
|---|---|---|
| Atari-HEAD (Zhang et al., 2019) | About 117 h of expert human play over about 20 Atari games: frames, actions, eye gaze | Good: human play at full resolution |
| JAT dataset (`jat-project/jat-dataset` on HF) | Expert trajectories for the 57 Atari games and more | Easy to load; frames are low-res grayscale |
| DQN Replay Dataset (Agarwal et al., 2020) | Millions of frames per game from DQN training, 60 games | Large; 84×84 grayscale; mixed-quality play; better for offline RL |
| CS:GO behavioural cloning (Pearce & Zhu, 2021) | Millions of Counter-Strike frames with keyboard and mouse actions | 3D FPS like Doom; needs actions mapped to a choice list |
| MineRL / OpenAI VPT contractor data | Minecraft video with keyboard and mouse | Large; long-horizon tasks |
| VideoGameBunny instruction data, GlitchBench | Question-answer pairs about game screenshots | Teaches reading game screens rather than acting; check contents and licences before use |

Check each licence; several are research-only or non-commercial.

## Practical notes

- **Motion needs two frames.** One frame can't show which way a ball, car or ghost is moving. `laya.vlm` accepts `{"images": [previous, current]}`, so training and play can both pass the last two frames.
- **Cover off-expert states.** If the data only comes from the expert, the model never sees the situations its own mistakes lead to. Mix in random actions when collecting (epsilon-expert, as below), or use DAgger: play the trained model, label what it saw with the expert, and retrain.
- **Soft labels over hard ones** whenever the expert has probabilities.
- **Watch forgetting.** Training only on one game may erode the photo-question skills. Mix in some VQA data, or score the VQA val splits after training (`modal run modal_app.py::evaluate`).
- **Calibration temperatures are per question type.** `finetune_long --init-from` keeps the starting checkpoint's temperature for any type missing from the new data. Game data is all `choice`, so the yes/no temperature is kept.

## ViZDoom `basic` with auto-labels (implemented)

`basic` is one room with one monster somewhere along the far wall. The player can only strafe left, strafe right or shoot. Reward is +106 for the kill, −5 per missed shot and −1 per tic, and the episode ends at 300 tics.

- **Expert** (`laya.games.doom_basic_expert`): if the monster's bounding box covers the crosshair column, ATTACK; otherwise strafe toward it. Over 100 episodes:
  - expert: mean reward +80.3, kills 100%
  - random: −134.6, kills 62%
  - always ATTACK: −218.2, kills 42%
- **Data** (`modal run modal_app.py::prepare_doom_basic`): 20k train and 2k val frames with disjoint seeds. Frames come from an epsilon-expert (30% random buttons), and every frame with a visible monster is labelled with the expert's button. It is written to `laya-datasets:/data/vqa/doom_basic/` in the same JSONL format as the VQA sets.
- **Training:** `modal run --detach modal_app.py::finetune_long --datasets doom_basic --init-from all3-3ep/best --run-name doom-basic --epochs 2`
- **Evaluation by playing** (`modal run modal_app.py::doom_eval --models all3-3ep/best,doom-basic/best`): 50 episodes per policy on unseen seeds. It reports mean reward, kill rate and the action mix for the expert, random, always-ATTACK, the zero-shot model and the trained model.
- **Watch it:** download `/ckpt/smolvlm/doom-basic/best` and run `python examples/vizdoom_live.py --model <that dir>`.

### Results (2026-09-19)

Data: 20,000 train frames (7,428 MOVE_LEFT, 7,124 MOVE_RIGHT, 5,448 ATTACK) and 2,000 val frames. Training: 2 passes from `all3-3ep/best` on one A100, 7.3 minutes including evaluation. The best checkpoint was step 924.

**Frame accuracy against the expert's labels (val):** 95.9% after 0.5 pass, 98.8% after 1, 99.5% after 1.5 and 2. ECE was 0.005 raw and 0.011 calibrated.

**Playing** (50 episodes on unseen seeds, `doom_eval` / `play_doom`):

| Policy | Mean reward | Kill rate | Steps per episode |
|---|---|---|---|
| Scripted expert | +75.8 | 100% | 6.8 |
| **Trained model (`doom-basic/best`)** | **+75.4** | **100%** | 6.8 |
| Random buttons | −121.9 | 68% | 39.4 |
| Always ATTACK | −325.6 | 18% | 63.0 |
| Zero-shot model (`all3-3ep/best`) | −325.6 | 18% | 63.0 (chose ATTACK on all 3,151 steps) |

The trained model plays at expert level.

**Cost: some forgetting of the photo tasks** (full VQA val splits):

| | Before | After Doom training |
|---|---|---|
| A-OKVQA acc | 61.8% | 57.9% |
| ScienceQA acc | 86.6% | 83.0% |
| VQAv2 yes/no acc | 73.4% | 71.6% |
| VQAv2 yes/no ECE (calibrated) | 0.041 | 0.105 |

The refit `choice` temperature (8.30, up from 3.33) is shared by all `choice` questions, so it now also flattens the photo multiple-choice answers. The yes/no temperature was kept at 1.69, but the model underneath changed, so yes/no calibration got worse. Mixing VQA data into the game training, or fitting temperatures per task, should fix both.

## Atari, game-only model (implemented, 2026-09-19)

A model trained on Atari only: plain SmolVLM-256M with a fresh head, and none of the photo datasets. The data format is in [atari-data-format.md](atari-data-format.md). All sources are on `laya-datasets:/data/atari/<source>/<Game>/`.

| Source | Games | Frames per game | Format | Labels | Code |
|---|---|---|---|---|---|
| `expert`: CleanRL PPO agents (best of 9 checkpoints per game) | 57 | 20k train / 1k val | RGB 210×160 | agent's action probabilities (soft) | `laya/atari_data/expert.py`, `modal_atari_expert.py` |
| `jat`: `jat-project/jat-dataset` (Apache-2.0) | 57 | 20k / 1k | gray 84×84, newest frame of each stack | agent actions (hard) | `laya/atari_data/jat.py`, `modal_atari_jat.py` |
| `atari_head`: Atari-HEAD v4, Zenodo 3451402 (CC-BY-4.0) | 20 | 20k / 1k | RGB 210×160 | human actions (hard) | `laya/atari_data/atari_head.py`, `modal_atari_head.py` |

Checks behind the data:
- **Action mapping:** JAT and the expert agents use ALE's minimal action set; Atari-HEAD uses the full 18-action enum. Each was verified against the data, not just the docs.
- **JAT KungFuMaster and MontezumaRevenge** store frame stacks in a different byte layout. The converter decodes them and checks every game's frame order before writing.
- **Expert baselines** are in each expert `meta.json`, measured with sticky actions: uncapped, and capped at 4,500 decisions (`*_cap4500`, which evaluation uses). Solaris's expert scores below random, so Solaris is excluded from summaries.

**Training** (`modal_atari_train.py::train_atari`; vision tower frozen, full LM and head; up to 4k frames per source per game; best checkpoint by val NLL):

| Run | Data | Steps | Val frame accuracy | Calibrated ECE |
|---|---|---|---|---|
| `atari-all-v1` | 122 source×game sets, 465k frames, 0.9 pass, 70 min A100 | 13,272 | 9.6% → 33.5% | 0.056 |
| `atari-expert-v1` | expert only, 45 games, 0.8 pass, 25 min | 4,315 | 9.2% → 30.4% | 0.087 |

**Playing** (`atari_eval`: ALE v5 defaults, auto-FIRE, 3 episodes × 4,500 decisions per game). Normalised score = (model − random) / (expert − random), with capped baselines:

| Model, action choice | Median normalised (unflagged games) | Beats random |
|---|---|---|
| `atari-expert-v1`, top action | 0.002 | 23/45 |
| `atari-expert-v1`, sampled | 0.010 | 31/45 |
| `atari-all-v1`, top action | −0.003 | 21/57 |
| `atari-all-v1`, sampled | 0.002 | 32/57 |

**Result: roughly random-level play.** A few games show real skill:
- Freeway 0.63 (always go UP)
- Centipede 0.20–0.34
- Robotank 0.23
- Krull 0.19–0.23

Adding JAT and Atari-HEAD didn't help: expert-only was at least as good on play and on expert-frame NLL. Grayscale 84×84 JAT frames don't match the RGB frames seen during play.

Flagged games, whose normalised scores aren't skill:
- **Skiing:** the expert just presses NOOP.
- **DoubleDunk, Tennis:** stalling until the cap scores well.
- **Pitfall, PrivateEye, MontezumaRevenge:** expert ≈ random.
- **Tutankham:** the greedy expert gets stuck.

With top-action play the model presses nearly one button, so the 3 episodes coincide even with different seeds.

Why it falls short, and what to try:
- **Single frame, no motion:** use two-frame input.
- **Under one pass at 4k frames per game:** train on expert-only data with all 20k frames and several passes.
- **Compounding errors in pure imitation:** use DAgger.
- **Frozen photo-trained vision tower on pixel art:** unfreeze the top vision layers.
- **57 very different games in one small model:** try specialists or small game groups.

## Atari RL fine-tuning on one game's score (implemented, 2026-09-19)

`laya/atari_rl.py` and `modal_atari_rl.py`: PPO on `ALE/Breakout-v5` score, starting from the single-frame
imitation checkpoint `atari-8g-1f/best`. The policy trained is exactly the one `laya.atari_train.play` uses --
the checkpoint's option distribution divided by its calibrated `choice` temperature (1.227 here) -- so a
checkpoint trained this way plays unchanged. What makes it affordable for a VLM policy: the prompt is built
once per game, each frame's image features are computed once by the frozen vision tower and then replayed
through the language model only, and the KL reference shares those same features.

Run `atari-rl-breakout-v1`: 128 envs in 2 lanes x 32 steps = 4,096 decisions per iteration, 2 epochs,
minibatch 128, clip 0.1, entropy bonus 0.01, KL against the starting policy adapted towards 0.3 nats, last 8
LM layers plus head plus a fresh value head trainable (36.8M params), reward clipped to +-1, episodic life.
**52 iterations, 213k decisions (about 850k emulator frames), 32 A100-minutes at ~145 decisions/s.** The best
checkpoint is chosen by the rolling mean of the last 30 full-game raw scores from the rollout itself, not by a
loss: iteration 49, mean 10.07.

**Playing** (`modal_atari_rl.py::evaluate`, 10 episodes, ALE v5 defaults with sticky actions, auto-FIRE,
4,500-decision cap). Normalised = (score - random) / (expert - random) using the `*_cap4500` baselines from
`/data/atari/expert/Breakout/meta.json` (random 1.2, expert 218.0):

| Policy | Score | Normalised | Mean steps | Action mix |
|---|---|---|---|---|
| Random | 0.8 | ~0.00 | - | - |
| Imitation start, sampled | 3.5 | 0.011 | 241 | NOOP .28 FIRE .26 LEFT .24 RIGHT .21 |
| Imitation start, top action | 11.4 | 0.047 | 499 | FIRE .42 NOOP .27 LEFT .18 RIGHT .14 |
| **RL-tuned, sampled** | **8.5** | **0.034** | 368 | LEFT .30 NOOP .28 FIRE .21 RIGHT .21 |
| **RL-tuned, top action** | **16.3** | **0.070** | 480 | NOOP .46 LEFT .31 RIGHT .12 FIRE .11 |
| Expert (CleanRL PPO) | 218.0 | 1.00 | - | - |

Paired by seed (both policies see the same 10 seeds): top action +4.9 (exact sign-flip permutation test
p = 0.10), sampled +5.0 (p = 0.03). So the sampled gain is solid at 10 episodes and the top-action gain is
suggestive but not significant; more episodes would settle it.

**RL fine-tuning works, and the gain is interpretable.** The imitation policy spent 42% of its top-action
presses on FIRE; with auto-FIRE handling the launch, FIRE is a wasted press once the ball is in play. RL cut
it to 11% and moved that budget into paddle movement (LEFT .18 to .31). Episodes also got longer for the
sampled policy (241 to 368 decisions), so the paddle really is surviving longer rather than just scoring
luckier.

**No action collapse.** Action entropy stayed at 1.97-1.99 bits of a possible 2.00 throughout training, and the
evaluated mix uses all four actions (1.76 bits top-action, 1.98 sampled). Policy entropy moved 1.27 to 1.17
nats against a ln 4 = 1.386 ceiling, and KL against the starting policy only reached 0.11 nats -- `kl_coef`
adapted down to its 0.001 floor by iteration 13 and stayed there, so the anchor was never the binding
constraint. The policy moved as little as it did because of compute, not because it was held back.

**But it is not the cheapest gain available.** 213k decisions is about two orders of magnitude short of a
standard Breakout PPO curve (10M frames, roughly 2.5M decisions at frameskip 4, which is where CleanRL reaches
~400), and at 145 decisions/s that curve is ~4.8 A100-hours for one game -- more than the 70 A100-minutes the
whole 57-game imitation model cost. The value function also never got traction (explained variance plateaued
at 0.28-0.36), which is what you would expect when the policy cannot see which way the ball is moving: one
frame does not contain the ball's velocity, so no amount of policy-gradient signal can recover it.

### If we return to RL, it has to be two-frame

Single-frame training is retired, and Breakout is the clearest case why. What a two-frame version needs:

- **An init checkpoint in the same format.** `atari_rl` can only fine-tune a policy whose input format matches
  what it feeds; `atari-8g-1f/best` is single-frame. RL waits on a two-frame imitation checkpoint trained on
  `expert2f` data.
- **The rollout has to carry the previous frame.** `VecAtari` keeps one `obs` per env; it needs a `prev_obs`
  alongside it, and `PromptTemplate` / `FramePreprocessor` have to be built from `{"images": [prev, cur]}`
  instead of `{"image": frame}`. Decide explicitly what `prev` is on reset and after a lost life (duplicate the
  post-FIRE frame is the obvious choice) and match whatever the two-frame data did, or training and play
  disagree.
- **Throughput roughly halves.** Preprocessing is already the CPU-bound half that the lanes hide behind the
  GPU, and it doubles; the vision tower doubles; the prompt grows by one 64-token image block (about +35-45% of
  the sequence), so the LM cost rises too. Expect ~70-90 decisions/s rather than 145, and raise `n_lanes` and
  `prep_threads` to keep preprocessing hidden.
- **Re-run `bench` first.** It asserts the cached-feature fast path reproduces `laya.atari_train.action_probs`.
  With two images the marker positions and sequence change, and that assertion is the only thing guaranteeing
  the policy being trained is the policy being evaluated.
- **Cheap early diagnostic:** explained variance should break past 0.36 once the ball's direction is
  observable. If it does not, the value head, not the input, is the problem.

Operational note: `best/` is now written about every ten minutes during a run, not only at the end, and
`modal run modal_atari_rl.py::recover --run-name <run>` rebuilds it from `state.pt`. The first run of this
experiment was killed early and left only `state.pt`, so nothing was evaluable until the weights were
extracted by hand.

## Next ideas

1. **Atari with SB3 teachers.** Start with Freeway and Breakout: log each teacher's action probabilities on full-colour frames and use them as soft targets.
2. **Two-frame input** for games where motion matters: Pong, Breakout, Freeway.
3. **DAgger rounds** on Doom `defend_the_center` (turn plus shoot), with the labels buffer as the expert.
4. ~~**RL fine-tuning** from game reward~~ -- done on Breakout (above): a real but small gain for A100-hours
   that imitation spends better. Revisit only on two-frame input.
5. **Mixed multi-game training** with VQA data mixed in, then checking both play strength and VQA accuracy.
