# Testing System One: how to run a real experiment

This is a guide to measuring Juno's System One yourself and writing it up
honestly. It covers what the claim is, what to compare it against, the exact
commands, which numbers to report, and what can make the results misleading.

Every command runs from the repo root, inside the project's virtualenv
(`source .venv/bin/activate`, or prefix with `.venv/bin/`).

---

## 1. The claim, stated so it can fail

> A small audio-only model (System One), trained by distillation from the
> transcribe-then-read pipeline (System Two), can decide **who an utterance
> was for** and **what was wanted** well enough to skip speech-to-text on a
> meaningful share of utterances. It does this without more false
> activations or missed requests than always transcribing, and with lower
> latency and compute.

Break it into hypotheses, each with a number that would refute it:

| | Hypothesis | Refuted if (suggested bar) |
|---|---|---|
| H1 | System One avoids STT on a meaningful share of utterances | `stt_avoided` < 30% at the budgets below |
| H2 | It stays safe | false-ignore > 1% or false-activation > 1% on held-out data (watch the 95% upper bound, not just the point estimate) |
| H3 | End to end it is no worse than always transcribing | hybrid correctness more than 1 point below the best always-transcribe baseline |
| H4 | Typed commands get faster | typed-command p50 latency not lower than always-transcribe with **Parakeet** (the fast baseline, not Whisper) |
| H5 | Distillation is what makes it work | a student trained on teacher labels is no better than one trained on nothing but gold labels, or than the no-pretraining encoder |
| H6 | The pretrained encoder matters | the `logmel` baseline matches the Parakeet encoder |

H3 and H4 decide whether this is a product. H5 and H6 decide whether it is a
*finding*.

---

## 2. What to compare against

Run every system on **the same held-out clips, on the same machine, in the
same session**. `bench` does this for you.

### Baselines (always transcribe)

| Name | Why it's there |
|---|---|
| `always:whisper:small.en` | Juno's default today. The "before" picture. |
| `always:whisper:large-v3-turbo` | Accuracy reference: the strongest local recogniser. |
| `always:parakeet-110m` | **The baseline that matters.** STT is already cheap here (~45 ms on an M1). If System One only beats Whisper, the result is about Whisper being slow, not about skipping STT. |
| `always:parakeet-0.6b` | Strong and still fast. Also the default teacher. |

### Prior "skip STT" approaches already in this repo

| Name | How to get it |
|---|---|
| Rule gate | `intent.gate.mode: skip`, rules only. It skips about 3% of real-log traffic (README, "The numbers"). |
| Learned feature gate | The `gate` encoder: the same 20 features the learned gate reads, under the same student, splits and thresholds as System One. It's the honest "simple model" baseline, and unlike `gate_training` it's trained on exactly the same rows. |

### Ablations of System One itself

Change one thing at a time, keep the split and seed fixed:

| Knob | Values | Command change |
|---|---|---|
| Encoder | `parakeet` (17 layers), `whisper` (small.en encoder), `logmel` (no pretraining), `gate` (the gate's 20 hand-made features) | `embed --encoder parakeet whisper logmel gate` |
| Encoder depth (early exit) | Parakeet at 4 / 8 / 12 / 17 layers | `--encoder parakeet:mlx-community/parakeet-tdt_ctc-110m@L8:stats` |
| Pooling | `stats` vs `stats_thirds` (keeps rough order) | `...@L17:stats_thirds` |
| Head | linear vs one hidden layer | `train --hidden 0` / `--hidden 256` |
| Targets | teacher (distillation) / gold / mix | `train --targets teacher|gold|mix` |
| Teacher strength | Parakeet 110M, Parakeet 0.6B, Whisper large-v3-turbo, with/without the LLM second opinion | `label --stt ... [--llm anthropic]` |
| Training data | synthetic only / + AMI / + Speech Commands / + your consented sessions | which manifests you pass to `label` and `train` |
| Budgets | 0.5%, 1%, 2% | `--max-false-ignore`, `--max-wrong-act`, `--max-false-activation` |
| Threshold margin | point estimate vs. Wilson bound at z = 1, 2 | `--margin-z 0 / 1 / 2` |
| Corpus mix | more commands, more hard negatives | `synth --mix typed=0.6,open=0.1,human=0.15,hard_negative=0.1,background=0.05` |

### External reference points (cite them, don't claim to beat them)

- **Apple, "Device-Directed Speech Detection: Regularization via
  Distillation"** ([arXiv 2203.15975](https://arxiv.org/abs/2203.15975)).
  Distilling an ASR-based model into an acoustics-only one gave a 66% relative
  EER improvement. That's the closest published analogue to the
  "for Juno?" half.
- **SLURP** ([EMNLP 2020](https://aclanthology.org/2020.emnlp-main.588/)):
  the standard end-to-end SLU benchmark, ~90% intent accuracy at the state of
  the art. Its audio is CC BY 4.0. Writing a Juno schema from SLURP's intents
  and running `train`/`evaluate` on it gives you a number that's directly
  comparable to published work. That's the strongest external check you can
  add.

---

## 3. Data

### What to build it from

```bash
# 1. Synthetic: scripted lines voiced by macOS TTS, plus augmented copies
#    (reverb, noise, level, far-field) and non-speech clips. Takes ~1 line/s.
python -m juno_core.slu synth --out data/slu/syn --size 3000 --augment 1 --noise 150

# 2. Real people talking to each other, far-field (AMI, CC BY 4.0, ~45 MB/meeting)
python -m juno_core.slu import-ami --out data/slu/real

# 3. Real voices saying "stop" / "yes" / "no" (Speech Commands test set, CC BY 4.0, 112 MB)
python -m juno_core.slu import-speech-commands --out data/slu/real

# 4. YOUR voice in YOUR room -- the most important set for the report
python -m juno_core.slu collect --out data/slu/consented/s1 --speakers p1,p2 \
    --room kitchen --consent release
```

Or do step 4 in a browser: `python -m juno_core.slu studio`, then the
**Collect** tab. Sessions land in `data/slu/studio/` in the same format, and
the **Results** tab evaluates and trains on them. The studio's **Try it** tab
is also the quickest qualitative check: speak, and watch System One's
decision next to System Two's for every utterance.

Nothing writes audio except the TTS engine itself. Augmented copies, AMI
utterances, and noise clips are recipes in the manifest, rebuilt in memory.
`collect` keeps encoder vectors and the teacher's labels, never audio.

### Labelling with the teacher

```bash
python -m juno_core.slu label --stt parakeet-0.6b --aliases "Jumo,June" \
    --manifest data/slu/syn/manifest.jsonl data/slu/real/ami.jsonl \
               data/slu/real/speech_commands.jsonl \
    --out data/slu/labelled.jsonl
```

Every row now carries both the gold label (from the script or the corpus)
and the teacher's soft answer. That's what makes H5 testable, and it's
what lets you report **the teacher's own accuracy** (`teacher_vs_gold` in
every report). Do report it: the student can't be expected to beat a
teacher it only learned from.

### Splits

`train` splits **by speaker** (TTS voice, AMI meeting, Speech Commands
speaker) into train / select / holdout (60/20/20). Thresholds are chosen on
*select*, and *holdout* is touched once. The model file records its holdout
ids, and `evaluate --only-holdout` and `bench --only-holdout` use exactly
those.

For the report, also hold out data **the model never saw in any form**:

- a second synthetic set with a different seed (`synth --seed 7`): new
  random fills, but the same templates. See threats, below.
- a consented session recorded on a different day or in a different room;
- (best) SLURP or another public set.

---

## 4. The runs

```bash
# vectors for every encoder you want to compare (cheap; once per dataset)
python -m juno_core.slu embed --rows data/slu/labelled.jsonl \
    --encoder parakeet logmel whisper \
              parakeet:mlx-community/parakeet-tdt_ctc-110m@L8:stats \
    --out data/slu/emb

# the whole ablation grid in one go: every encoder table found, three target
# modes, three seeds -> reports/sweep/sweep.md (mean ± sd) and sweep.csv
python -m juno_core.slu sweep --rows data/slu/labelled.jsonl --emb-dirs data/slu/emb \
    --targets teacher mix gold --seeds 0 1 2 --allow-internal --out reports/sweep

# or one student at a time
python -m juno_core.slu train --rows data/slu/labelled.jsonl \
    --embeddings data/slu/emb/parakeet__mlx-community_parakeet-tdt_ctc-110m_L17__stats.npz \
    --targets mix --allow-internal --out models/s1-parakeet-mix.npz \
    --report reports/s1-parakeet-mix.holdout.json

# the live benchmark: every system on the holdout clips, timed end to end
python -m juno_core.slu bench --rows data/slu/labelled.jsonl --only-holdout \
    --model models/s1-parakeet-mix.npz --fallback-stt parakeet-110m \
    --baselines parakeet-110m parakeet-0.6b whisper:small.en whisper:large-v3-turbo \
    --out reports/bench.json --csv reports/bench.csv
```

`--allow-internal` is needed while the training set contains macOS TTS
audio, whose licence doesn't allow redistributing derived models. The model
file then says `"distributable": false`. That's fine for an experiment;
it's not fine for publishing weights.

**Repeat with 3 seeds** (`sweep` does this; with `train`, `--seed 0/1/2`
changes the split) and report the mean ± standard deviation. A single split of a few hundred
clips can move a rate by several points.

### Energy

`bench` reports wall and CPU time. GPU energy needs root:

```bash
sudo powermetrics --samplers cpu_power,gpu_power -i 200 -o power.txt &
python -m juno_core.slu bench ... --baselines parakeet-110m          # one system per run
sudo pkill powermetrics
```

Integrate `CPU Power` + `GPU Power` (mW) over the run and divide by the
clip count to get **mJ per utterance**. Measure an idle baseline the same
way and subtract it. Run each system separately (`--baselines` with one
entry, or `--baselines` empty plus `--model`) so the windows don't overlap.

### In the wild: shadow mode

The offline numbers come from clips. The real test is days of your own use:

```yaml
system_one:
  mode: shadow
  model_path: models/s1-parakeet-mix.npz
```

Every turn then logs System One's decision next to System Two's, and
nothing changes for you. Afterwards:

```bash
python -m juno_core.slu shadow --log logs/events.jsonl
```

That gives you the agreement matrix, how many turns would have skipped STT,
and every disagreement that matters: `would_have_missed`,
`would_have_acted_on_ignored`, `would_have_acted_differently`. This is the
most convincing evidence you can put in a report, because nobody staged it.
Note that System Two is the reference here, not ground truth. Hand-check
the disagreements.

---

## 5. What to report

### Headline table (one row per system, held-out clips)

| System | Correct | False activation (95% upper) | Missed | STT calls avoided | p50 / p95 latency (ms) | Typed-command p50 (ms) | Wall s / CPU s | mJ / utt |
|---|---|---|---|---|---|---|---|---|
| always: whisper small.en | | | | 0% | | | | |
| always: whisper large-v3-turbo | | | | 0% | | | | |
| always: parakeet-110m | | | | 0% | | | | |
| always: parakeet-0.6b | | | | 0% | | | | |
| System One + parakeet-110m | | | | | | | | |

All of these come from `bench.json` → `systems.*`.

### Student quality (from `train --report` / `evaluate --full`)

- routes (act / ignore / escalate), `stt_avoided`
- `false_ignore`, `false_activation`, `wrong_act`, each with its 95% upper bound
- `addressed_accuracy`, `addressed_auc`, `addressed_ece` (calibration)
- `intent_accuracy`, `intent_ece`, `slot_accuracy`, the top `intent_confusions`
- `typed_commands_acted`: the share of typed commands handled with no STT
- `teacher_vs_gold`: the teacher's own accuracy, false activations, misses
- `by_source` and `by_category`: **check that real-audio rows (`ami`,
  `speech_commands`, `consented`) aren't much worse than synthetic ones**
- `encode_ms` p50/p95, model size (`ls -la` the `.npz`), parameter count

### Plots worth making

1. **Coverage vs. risk.** Sweep `--max-false-ignore` / `--max-false-activation`
   (0.25% to 5%) and plot `stt_avoided` against the measured false-activation
   rate. This is the trade-off in one picture.
2. **Reliability diagram** for p(assistant) and p(intent): binned confidence
   vs. accuracy. (`addressed_ece` summarises it.)
3. **Latency CDF** per system from `bench.csv` (`ms` column), all clips and
   typed commands only.
4. **Encoder depth vs. accuracy vs. encode time** (Parakeet L4/L8/L12/L17).
5. **Ablation bars:** teacher vs gold vs mix targets; logmel vs whisper vs
   parakeet encoder.

---

## 6. Threats to validity (put these in the report)

- **Synthetic speech isn't speech.** TTS voices are cleaner and more
  regular than people, and the people in the room don't speak to a TTS
  engine the way they speak to a device. Real-audio rows and consented
  sessions are the check. Report `by_source`.
- **Templates leak.** Synthetic train and test share phrasing templates,
  even across seeds. The split holds out *voices*, not *wordings*. A
  student can look better on synthetic holdout than it is on new phrasings.
  Consented sessions and SLURP are the check.
- **The teacher's parser was written alongside the corpus.** On clean text
  it agrees with the script 100%, so `teacher_vs_gold` errors on synthetic
  data come from the recogniser and the addressee engine, not the parser.
  Real speech will hit phrasings the parser doesn't know, and those become
  `open_request` (safe: they escalate).
- **Cold-start evaluation.** Clips have no conversation history, so
  intents bound to a state (`confirm.yes` needs an open question) always
  escalate offline. The addressee engine is also at its most conservative
  cold: it misses many short commands it would accept inside a conversation.
  `evaluate --awaiting-answer` shows the other extreme. Shadow mode shows
  reality.
- **Minimal pairs.** Pooling an utterance into one vector loses short
  sounds that flip the meaning. In the reference run, every wrong act was
  "unpause" heard as `media.pause`. Build a minimal-pair test set ("pause /
  unpause", "turn it on / off", "louder / not louder") and report it
  separately; set `min_confidence` on such intents in your schema.
- **Slots are data-starved.** A timer's length is one of 23 classes, learned
  from a few hundred clips, and was right about a quarter of the time. The
  router escalates rather than guess, but test whether
  `synth --mix typed=0.6,...` (more command data) or more timer phrasings
  move `slot_accuracy`.
- **Truth for real corpora is partial.** AMI is labelled `human_directed`
  by construction (people in a meeting). Speech Commands words are labelled
  as commands to the assistant, which is an assumption.
- **One machine, one run order.** Thermal state and background load move
  latency. Interleave systems, or repeat runs and report the spread. Report
  the chip (`bench.json` → `machine`).
- **Licences.** A model trained on macOS TTS audio is not distributable.
  Say so if you publish numbers, and don't publish those weights.

---

## 7. A short checklist

- [ ] Data built, labelled, embedded; `sources` shows the provenance
- [ ] Teacher accuracy reported (`teacher_vs_gold`)
- [ ] 3 seeds per configuration; mean ± sd
- [ ] Bench on holdout against all four always-transcribe baselines
- [ ] Ablations: encoder, depth, targets, head
- [ ] Coverage-vs-risk curve
- [ ] At least one consented session, recorded on a different day from training
- [ ] A week of shadow mode, disagreements hand-checked
- [ ] Energy measured with powermetrics (or explicitly left out)
- [ ] Threats section written

---

## 8. What the reference run found (a starting point, not a result)

One run on an Apple M1, October 2026. Data: 6,148 synthetic clips (30 macOS
voices, half augmented), 562 AMI far-field utterances, 360 Speech Commands
clips. Teacher: Parakeet 0.6B, the intent engine, and the parser, with no
language-model second opinion. Splits hold out voices. Reproduce it with the
commands above; the sweep table is `reports/sweep/sweep.md`.

| | Verdict | Evidence |
|---|---|---|
| H1 coverage | **supported on this data** | 48% ± 3 of STT avoided (sweep, Parakeet 17 layers, gold); 47% in the live bench |
| H2 safety | **mostly** | student false-ignore 0.4%, false-activation 0.0%; wrong acts 1.1% ± 1.1%, all of seed 0's being "unpause" → `media.pause` |
| H3 accuracy | **supported, with a caveat** | hybrid 70.0% correct vs 66.7% for the best always-transcribe (Whisper turbo). The caveat: template overlap between train and test. |
| H4 latency | **split** | decided-by-System-One clips: 16 ms p50 vs 36 ms (Parakeet 110M) / 315 ms (Whisper small). But typed-command p50 overall is *not* lower than always-Parakeet, because over half of them escalate and pay both. Compute per utterance: −33% vs Whisper small, −8% vs Parakeet 110M. |
| H5 distillation | **refuted with this teacher** | teacher-only targets: 1% avoided, AUC 0.74. The cold teacher misses ~half of commands. Mix: 34%; gold: 48%. |
| H6 pretrained encoder | **supported** | log-mel 7%, gate features 6%, Parakeet 4 / 8 / 17 layers: 11 / 26 / 48% |

The experiments most likely to change the picture, roughly in order:

1. **A stronger teacher.** `label --llm <provider>` (the second opinion on
   every ambiguous clip) and/or `--stt whisper:large-v3-turbo`. Re-test H5:
   does pure distillation work once the teacher is good?
2. **Consented real sessions** (`collect`) as a held-out test set, recorded
   on a different day. This is the number that matters most.
3. **SLURP** with a schema built from its intents, for a number comparable
   to published end-to-end SLU.
4. **Minimal pairs** as a separate test set, and `min_confidence` on the
   intents involved.
5. **Better pooling for slots** (attention pooling over frames instead of
   mean/std/max), or more timer data via `synth --mix`.
6. **Speculative STT**: start transcribing in parallel when System One is
   unsure, and cancel it if the answer turns out to be *ignore*. This removes
   the escalation penalty in H4. It isn't implemented; the pipeline's
   docstring explains why head-start transcription was kept out of the
   reference wiring.
