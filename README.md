# Text-Free Fault-Ratio Estimation from Dashcam Video with Multi-Task Vision-Language Models

Official code, evaluation configurations, split definitions, and reported
prediction artifacts for:

> **Text-Free Fault-Ratio Estimation from Dashcam Video with Multi-Task Vision-Language Models**  
> Jiwoong Heo, Museon Choe, Jaehwan You, Jaedo Kwak, Jeongho Shin, and Duk-Jo Kong  
> _Submitted to IEEE Transactions on Intelligent Vehicles (T-IV), 2026._

This repository accompanies a text-free, role-structured multi-task
vision-language framework for guideline-based fault-ratio estimation from
first-person dashcam video. The framework jointly supervises counterpart
maneuver, fault comparison, and fault-ratio prediction in a single forward
pass. The ablations show that structured auxiliary supervision is the main
source of the observed gains, whereas explicit soft inter-head chaining does
not provide a consistent standalone accuracy advantage.


A LoRA-adapted vision–language backbone with three lightweight heads (situation → comparison → fault ratio) that replaces text-mediated cascades with a fully differentiable, single-forward-pass pipeline for insurance-grade traffic fault-ratio adjudication.

## Repository layout (numbered in paper order)

| Folder | Paper section | Contents |
|---|---|---|
| `code/1_pipeline_core` | III–IV | Frame extraction, backbone + triple-head model, training / deterministic evaluation (`agent1_pipeline.py`, `e2e_model3.py`, `e2e_train3.py`) |
| `code/2_main_experiments` | V-A/B/C/D | 18-run campaign, eval-mode re-evaluation, seed-robustness runner |
| `code/3_external_baseline` | V-B | VideoMAE-B baseline, inference-latency benchmark |
| `code/4_interface_supervision_ablations` | V-E | Hard-chain / no-chain / ratio-only variants (`E2E_CHAIN` switch) |
| `code/5_oracle_analysis` | V-F | Oracle-ladder computation |
| `code/6_zeroshot_diagnostic` | Supp. S-VIII | Zero-shot prompting re-scoring |
| `code/7_figures` | Figs. 3–4 | Qualitative-strip and confusion-matrix generation |
| `code/8_data_verification` | III-C/D | Label-discard and split-integrity checks |
| `splits/` | III-D | First-person split lists (file names only): train 7,574 / val 880 / test 2,843 / held-out 2,842 / reserved 1,769 |
| `results/` | Tables I–IV, Supp. | Eval-mode metrics (JSON) and per-clip predictions (CSV) for every reported run: 18 main configs, seed campaign (s44/s45), interface ablations (hard, no-chain ×3 seeds), supervision ablation (ratio-only ×3 seeds), oracle / zero-shot / VideoMAE artifacts |
| `figures/` | Figs. 3–4 | Camera-ready figure files |
| `docs/` | — | Numeric provenance: results summary, verification logs, evidence notes |

**Note on execution.** The runner scripts assume a flat working directory (as on our workstation): copy the contents of `code/1_pipeline_core` together with the runner you need into one directory, place the split lists beside them, and set `DATA_DIR` in `e2e_model3.py` to your local copy of the corpus. Interface variants are selected via the `E2E_CHAIN` environment variable (`soft` | `hard` | `none`).

## Data

Experiments use the publicly released **Traffic Accident Video corpus** from [AI Hub](https://aihub.or.kr) (NIA, Republic of Korea). Raw videos, frames, and label JSONs are **not redistributed** here under the corpus license; obtain them from AI Hub. The lists in `splits/` reference clips by file name and reproduce our exact partitions; the CSVs in `results/` contain per-clip predictions keyed by those file names.

## Reproducing the main results

```bash
bash run_paper_bb1.sh splits      # build first-person split files
bash run_paper_bb1.sh priority    # 18 training runs (4x GPU)
bash run_reeval_bb1.sh            # deterministic eval-mode re-evaluation
bash run_seeds.sh                 # seed-robustness campaign
E2E_CHAIN=none bash run_nochain_ms.sh    # interface ablation example
```

## Citation

```bibtex
@article{heo2026textfree,
  title   = {Text-Free Fault-Ratio Estimation from Dashcam Video with
             Multi-Task Vision-Language Models},
  author  = {Heo, Jiwoong and Choe, Museon and You, Jaehwan and
             Kwak, Jaedo and Shin, Jeongho and Kong, Duk-Jo},
  journal = {IEEE Transactions on Intelligent Vehicles},
  note    = {Under review},
  year    = {2026}
}
```

## License

Code is released under the MIT License (`LICENSE`). The AI Hub corpus and its annotations remain subject to their original terms.

## Acknowledgment

Supported by the InnoCORE program of the Ministry of Science and ICT (26-InnoCORE-01).
