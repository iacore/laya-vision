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

Why it falls short, and what to try (the first two are tested in the next section):
- **Single frame, no motion:** use two-frame input.
- **Under one pass at 4k frames per game:** train on expert-only data with all 20k frames and several passes.
- **Compounding errors in pure imitation:** use DAgger.
- **Frozen photo-trained vision tower on pixel art:** unfreeze the top vision layers.
- **57 very different games in one small model:** try specialists or small game groups.

## Two frames and more data per game, 8 games (implemented, 2026-09-19)

The two failures above (one frame, and under one pass over 4k frames per game) were tested directly on
**Breakout, Pong, Freeway, SpaceInvaders, Enduro, Boxing, Qbert and MsPacman**, with `/data/atari/expert2f/`
(every record carries `prev_image`, the frame from the previous decision step). Both runs start from
`atari-expert-v1/best`, use all 20k frames per game, the vision tower frozen, and differ only in `--frames`:

    modal run --detach modal_atari_train.py::train_atari --run-name atari-8g-2f --frames 2 --sources expert2f \
        --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman --passes 2 --max-minutes 55 \
        --init-from atari-expert-v1/best
    modal run modal_atari_train.py::atari_eval --model atari-8g-2f/best --episodes 10 \
        --games Breakout,Pong,Freeway,SpaceInvaders,Enduro,Boxing,Qbert,MsPacman

Two frames cost about 30% throughput (2.78 vs 3.86 steps/s at batch 32), so in 55 minutes the two-frame run
reached 1.84 passes against the one-frame run's 2.00.

**Val frame accuracy** (mean over games, calibrated): 0.31 at the start, 0.596 for two frames and 0.586 for one.
At matched passes two frames is ahead throughout, e.g. 0.575 / NLL 1.151 at 1.12 passes versus 0.554 / 1.187 at
1.20. Mean per-game val NLL: 1.098 (two frames) and 1.124 (one), both raw.

**Playing** (10 episodes per game, 4,500-decision cap, normalised with the `*_cap4500` baselines, top action):

| Game | expert | random | 2 frames | 1 frame | `atari-expert-v1` |
|---|---|---|---|---|---|
| Boxing | 92.4 | 1.6 | **0.57** | 0.37 | -0.05 |
| Freeway | 34.0 | 0.0 | **0.75** | 0.67 | 0.00 |
| Pong | 5.4 | -20.2 | **0.35** | 0.29 | -0.02 |
| Qbert | 24985 | 185 | **0.31** | 0.19 | -0.00 |
| MsPacman | 6092 | 434 | **0.10** | 0.07 | 0.00 |
| Breakout | 218.0 | 1.2 | 0.03 | 0.05 | 0.00 |
| SpaceInvaders | 8251 | 125 | 0.03 | 0.02 | 0.02 |
| Enduro | 379.6 | 0.0 | 0.02 | 0.04 | 0.00 |
| **median** | | | **0.201** | 0.131 | 0.000 |
| beats random | | | 8/8 | 8/8 | 3/8 |

Both new models beat random on every game, where the 57-game model beat it on 3 of 8. Two frames wins on 5 of 8
games and on the median (0.201 vs 0.131); one frame is marginally better on Breakout and Enduro. Sampling now
*hurts* (medians 0.091 and 0.089): once the policy is good, its own top action beats sampling from it, the
opposite of the 57-game model.

Fitted temperatures (about 1.22) made ECE worse, not better (0.094 raw to 0.118 calibrated for two frames): the
calibration holdout comes from held-out episodes of the same games, and the model is already slightly
underconfident there.

## Durability of long runs

Modal preempts containers, and an A100 training run is the expensive thing to lose, so `train_atari` is
resumable and `play_atari` / `train_atari` carry `retries=modal.Retries(max_retries=3, initial_delay=10.0)`:
a preempted container restarts and continues instead of dying.

- Every `--state-every-min` minutes (default 10) and after every eval, the run writes `<run>/state.pt`: weights,
  optimizer state, step, RNG states, per-group sample counts, the best-so-far record and the log. It is written
  to `state.pt.tmp` and renamed, then the volume is committed, so a torn write never replaces a good state.
- On start, an existing `state.pt` is resumed and logged (`resuming ... at step N`). `--restart` ignores it, but
  only for that call's first attempt (the marker records the Modal function-call id), so a retry of a restarted
  run still resumes rather than starting over.
- `--max-minutes` counts **training time across attempts**: the resumed loop backdates its clock by the elapsed
  time in `state.pt`, so a preempted run cannot spend twice its budget. `max_passes` likewise counts the samples
  already drawn per group. The sampler is an endless random stream, so a resumed run re-seeds it (seed + step)
  instead of replaying the same order.
- `--crash-at-step N` raises once at step N to exercise the path. Verified: the run trained to step 22, wrote
  `state.pt`, crashed at step 44, and the retried container resumed at step 22 with 0.7 minutes already counted
  and finished normally.
- `atari_eval --out results.json` writes each game's result as it lands and skips games already in that file
  (matching model, episodes, cap, sampling, frames and gate), so an interrupted evaluation only redoes what is
  missing; `--restart` ignores the file. `write_synthetic` skips a game that already has `_READY` unless
  `force=True`; the real data jobs live in the `modal_atari_{expert,head,jat}.py` apps.

## Next ideas

1. **Atari with SB3 teachers.** Start with Freeway and Breakout: log each teacher's action probabilities on full-colour frames and use them as soft targets.
2. **Two-frame input** for games where motion matters: Pong, Breakout, Freeway.
3. **DAgger rounds** on Doom `defend_the_center` (turn plus shoot), with the labels buffer as the expert.
4. **RL fine-tuning** from game reward, using the same proper-scoring policy-gradient term the loss already has.
5. **Mixed multi-game training** with VQA data mixed in, then checking both play strength and VQA accuracy.
