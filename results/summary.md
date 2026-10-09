# quiet-cortex — results summary

Object detection of ERD/ERS events on motor imagery EEG spectrograms (BCI Competition IV 2a, 9 subjects).
Every number below is taken from `results/tables/` (file given in brackets). Validation = session T (20 % trial
pool), test = session E (cross-session). Model selection uses validation only.

## Data and labels

- Kept trials after artifact rejection: 2328 (session T), 2368 (session E) (`notebooks/dataset_and_erd.ipynb`, `dataset/processed/metadata.csv`).
- Grand averages show the expected contralateral hand ERD (left hand: C4 −22.6 %, right hand: C3 −23.8 %, mean 8–30 Hz, 0.5–4 s); feet show only a brief early ERD at Cz followed by beta ERS, tongue shows power increases over the hand areas (`notebooks/dataset_and_erd.ipynb`).
- **Single trials cannot be labeled reliably (Q0).** With a ±20 % threshold, ERD candidates appear on every panel in ~85–97 % of trials regardless of the cue (`autolabel_variants_specificity.csv`). A z-score computed on per-trial baseline-corrected maps is inflated (post-cue / baseline SD ratio 2.4, `k1_label_null_sd_ratio.csv`). With the corrected session-level z, no class reaches 3× phase-randomized surrogates at k = 1 (C4 1.77×, C3 2.61×, Cz 1.45×, lateral 0.48×, ERS 2.17×; `k1_label_null_test.csv`).
- **Final unit: mean power of 5 trials** (same cue, subject, split); session-level dB z map, rendered and thresholded identically (|z| ≥ 3, ≥ 0.5 s, ≥ 2 Hz, largest component per panel, class = panel). Real vs. surrogate boxes per image: C4 22.2×, C3 13.1×, Cz 16.7×, lateral 11.5×, ERS 5.3×; background 56.2 % real vs 93.5 % surrogate (`kavg_label_null_test.csv`).
- Dataset: 1800 / 360 / 1800 images (train / val / test), background 56.1 / 57.2 / 50.7 %, 1538 / 309 / 1951 boxes (`kavg_label_stats.csv`). Mean pairwise trial overlap between groups: 9.7 % train, 38.7 % val, 7.7 % test (`kavg_group_overlap.csv`).
- Manual QC of 98 boxes in 50 training images: 82.7 % valid, 14.3 % unsure, 3.1 % invalid; ERD_C3 / C4 / Cz ≈ 94–95 % valid, ERD_lateral 66.7 %, ERS_rebound 68.2 %; box extent "good" for 44 of 98 boxes (`results/qc/qc_sheet.csv`, `notebooks/annotation.ipynb`).
- Cue agreement (labels never see the cue): the strongest ERD box matches the cue in 19.3 % (T) / 23.2 % (E) of groups vs. chance 7.6 % / 8.0 % (`kavg_cue_match_by_subject.csv`).

## E1 — training settings (YOLO11n, validation; `e1_hyperparams.csv`)

- Baseline (SGD, lr0 1e-2, batch 16, imgsz 640, 100 epochs, all augmentation off): mAP@0.5 0.974, mAP@0.5:0.95 0.815 → selected.
- Variants: epochs 50 0.973 / 0.790; imgsz 512 0.958 / 0.748; AdamW 0.956 / 0.773; lr0 1e-3 0.954 / 0.791; batch 32 0.953 / 0.813.
- Differences ≈ 0.02 mAP@0.5 are within what the small, overlapping validation set resolves; lower resolution clearly hurts box precision.
- Seeds 0 / 1 / 2 of the selected setting: val mAP@0.5 0.965 ± 0.007; test mAP@0.5 0.927 ± 0.005, mAP@0.5:0.95 0.740 ± 0.011 (`seed_runs.csv`).

## E2 — architecture (test; `e2_architecture.csv`, `per_class_ap.csv`)

| model | val mAP@0.5 | test mAP@0.5 | test mAP@0.5:0.95 | precision | recall | onset / offset error |
|---|---|---|---|---|---|---|
| Threshold rule (= label generator, reference) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 / 0 ms |
| YOLO11n (selected) | 0.974 | 0.925 | 0.735 | 0.812 | 0.970 | 18.0 / 20.9 ms |
| YOLO11s | 0.967 | 0.937 | 0.777 | 0.806 | 0.972 | 17.2 / 20.3 ms |
| Faster R-CNN (ResNet50-FPN v2) | 0.933 | 0.837 | 0.596 | 0.695 | 0.885 | 31.8 / 32.4 ms |

- Weakest class on test: ERD_lateral (AP50 0.80 YOLO11n, 0.86 YOLO11s).
- YOLO errors are almost all false positives (YOLO11n: TP 1892, FP 438, FN 59 at conf 0.25, IoU 0.5). Of 333 class-unmatched false positives, 236 do not overlap any ground-truth box (sub-threshold events not marked by the rule), 92 overlap partially (localization), 5 hit another class (`notebooks/detection_gallery.ipynb`).
- Faster R-CNN swaps position-defined classes (e.g. 87 ERD_C4 boxes predicted as ERD_C3; `confusion/e2_frcnn_test.csv`), plausibly because its RoI classifier lacks absolute image position.
- Per subject, YOLO11n test mAP@0.5 ranges 0.79–0.98 (`per_subject_detection.csv`).

## E3 — 2 vs 5 classes (test; `e3_classes.csv`)

- AP50 ERD_C4 / ERD_C3: 5 classes 0.965 / 0.975, 2 classes 0.955 / 0.976 (AP50-95 0.836 / 0.877 vs 0.796 / 0.856). No gain from fewer classes.

## E4 — robustness to signal-level noise (test mAP@0.5; `e4_robustness.csv`)

| model | clean | 30 dB | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|---|---|
| Threshold rule | 1.000 | 0.895 | 0.626 | 0.429 | 0.168 | 0.031 | 0.007 |
| YOLO11n | 0.925 | 0.859 | 0.595 | 0.396 | 0.157 | 0.037 | 0.005 |

- Same relative degradation (YOLO11n keeps 92.9 / 64.4 / 42.8 / 16.9 % of its clean mAP at 30 / 20 / 15 / 10 dB; threshold 89.5 / 62.6 / 42.9 / 16.8 %) → no robustness advantage of the trained detector (Q4).
- The small Laplacian makes nominal per-channel SNRs much harsher: effective SNR (noise within 4–40 Hz) averages 22.4 / 12.4 / 7.4 / 2.4 / −2.6 / −7.6 dB for nominal 30 / 20 / 15 / 10 / 5 / 0 dB (`e4_effective_snr.csv`).
- 10 / 5 / 0 dB are the concept's levels; 30 / 20 / 15 dB were added to show the onset of the degradation.

## E5 — decoding per 5-trial group (test; `e5_decoding.csv`, `e5_coverage.csv`)

| method | accuracy | κ | no decision |
|---|---|---|---|
| CSP + LDA | 0.718 | 0.624 | 0 % |
| Threshold rule | 0.203 | −0.063 | 68.2 % |
| YOLO11n | 0.164 | −0.115 | 67.3 % |

- Groups with an ERD box: 32.7 % (YOLO11n); accuracy when a decision is made: 0.501 (YOLO11n), 0.514 (YOLO11s), 0.637 (threshold); chance 0.25.
- Coverage by cue (YOLO11n): left hand 58.9 %, right hand 46.2 %, feet 20.7 %, tongue 5.1 % → tongue and feet imagery rarely produce a location-specific ERD.
- Per-subject CSP + LDA group accuracy 0.37–0.98 (`e5_decoding_per_subject.csv`). Group accuracies are not comparable with single-trial results in the literature.

## Limitations

- Labels are automatic: ERD_lateral and ERS_rebound are the least reliable classes (QC), ERS boxes can be broadband end-of-epoch artifacts, and box extent is often only the strongest core of a visible event.
- Classes are defined by panel position, so classification is trivial for one-stage detectors; the task is finding events and their extent.
- The threshold rule generates the labels, so its clean-data score is an upper reference, not a fair competitor.
- Validation is small (360 images) with 38.7 % trial overlap between groups; model selection is noisy (seed 0 was the best of three on validation).
- Images average 5 trials, so decoding results describe groups of trials, not single-trial BCI performance.
- Subjects with weak sensorimotor modulation (A02, A04, A05) contribute few or no ERD events (BCI inefficiency).

## Figures (`results/figures/`)

`grand_average_erd.png`, `single_trial_vs_group.png`, `label_validation_ratios.png`, `kavg_examples.png`, `box_size_distributions.png`, `cue_agreement_by_subject.png`, `pipeline.png`, `training_curves_e1.png`, `training_curves_e2.png`, `confusion_e2_test.png`, `e4_robustness.png`, `e5_decoding_by_subject.png`, `gallery_good.png`, `gallery_failures.png`, `gallery_tongue.png`, `noise_series.png`, `noise_series.gif`, `detections_A07_right_hand.gif`.
