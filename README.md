Project Overview

Milestone 1

Milestone 1 implements a deterministic pipeline for face verification experiments using the LFW dataset, including ingestion, pair generation, and similarity benchmarking. Ingestion downloads or loads LFW, applies a fixed train/validation/test split, and writes a manifest summarizing the dataset splits. Pair generation creates positive and negative verification pairs for each split, saved as CSV files. The similarity benchmark computes cosine similarity and Euclidean distance on synthetic vectors, comparing Python loop and vectorized implementations for speed and correctness. This milestone ensures reproducibility: running the scripts with the same configuration produces identical outputs.

Repository Layout:
The repository is organized with src/ containing the Python package code, scripts/ holding command-line scripts, and configs/ including YAML configuration files specifying seeds, paths, and split/pair policies. Generated artifacts such as manifests, pairs, and benchmark results are saved under outputs/ and are not committed. The data/ directory contains the dataset cache and is also ignored. Optional tests and determinism checks can be found in tests/. Committed files include the source code, scripts, configurations, README, and requirements; outputs and dataset caches are ignored.

How to Run:
To reproduce the milestone, first clone the repository and set up a Python virtual environment:

git clone https://github.com/gabebekele/msml605project <new_repo_name>
cd <new_repo_name>
python3 -m venv .venv

Activate the virtual environment:

for macOS/Linux:
source .venv/bin/activate

for Windows:
.venv\Scripts\Activate.ps1

And then install the requirements:

pip install -r requirements.txt

Next, run deterministic ingestion with:

python scripts/ingest.py --config configs/m1.yaml

This produces outputs/manifest.json, which summarizes the dataset splits, counts, seed, and split policy. Then, generate verification pairs using:

python scripts/make_pairs.py --config configs/m1.yaml

This produces outputs/pairs/train.csv, outputs/pairs/val.csv, and outputs/pairs/test.csv. Finally, run the similarity benchmark with:

python -m scripts.bench_similarity

The benchmark prints timing and correctness results for Python loop versus vectorized implementations, and saves them to outputs/bench_results.txt.


Outputs:
The outputs of this milestone include the manifest file summarizing splits and counts, three CSV files containing verification pairs for train, validation, and test sets, and benchmark results for similarity computations. All outputs are stored under the outputs/ directory.

Determinism Notes:
A fixed seed (123) is used to ensure determinism. The pipeline is deterministic because identities and filenames are sorted before splitting, all random operations use a fixed seed via np.random.default_rng(seed), and the pair generation policy is consistently applied. Rerunning ingestion and pair generation with the same configuration in order to produce identical outputs.


Milestone 2

Milestone 2 makes the deterministic pipeline into an iterative ML system. This stage transitions from synthetic benchmarks to a configuration-driven reproducible backbone capable of tracking multiple runs, managing normalization strategies, and performing automated error analysis. The system evaluates the model's ability to distinguish individuals using pairs of images and declare them the same and different people using Cosine Similarity on raw pixel intensities.

Experimental Summary
The evaluation involved a five-run sweep to isolate the impact of data preprocessing and threshold selection rules. The Baseline (Run 1) used raw pixel intensities without normalization, resulting in a Balanced Accuracy of 0.495. The Data-Centric Improvement implemented Z-score normalization, which provided the most significant performance gain, achieving a Balanced Accuracy of 0.5875. Analysis revealed that while normalization improved specific threshold performance, the model remains sensitive to environmental noise such as lighting, skin tone, and occlusions.

How to Run
To reproduce the evaluation and error analysis, ensure your virtual environment is activated and requirements are installed as described in Milestone 1.

The evaluation system uses a base configuration file with command-line overrides for specific experimental runs. To reproduce the results of the best run which used Z-score normalization and the balanced accuracy rule, execute:

python -m src.evaluation --run_id run_3_zscore --norm z-score

This command performs a threshold sweep, identifies the optimal decision boundary, and generates the corresponding metrics and visualizations. You can also reproduce the other runs like the Run 1 baseline or the Run 5 edge case by adjusting the flags. 

# Run 1: Baseline (No Norm)
python -m src.evaluation --run_id run_1_baseline

# Run 2:
python -m src.evaluation --run_id run_2 --metric f1

# Run 4:
python -m src.evaluation --run_id run_4 --norm min_max

# Run 5:
python -m src.evaluation --run_id run_5 --norm z-score --metric f1

To verify the evaluation metrics are wworking as intended run the metric test as shown below:

pytest tests/test_metrics.py

Outputs
Detailed artifacts for each experiment are stored in unique timestamped directories under outputs/runs/. The results include performance metrics in run.json containing accuracy, precision, and recall, as well as visualizations like roc_curve.png which illustrates the model's discriminative power. Categorized error slices are provided in the errors/fp/ and errors/fn/ subdirectories for qualitative analysis of false positives and false negatives. Finally, the formal evaluation and hypothesis report is located at reports/M2_report.pdf.

Reproduction Notes
To reproduce the selected threshold of 0.2653, ensure the configuration utilizes norm: z-score and a pair_limit: 200. Note that using the f1-score selection rule as seen in Run 5 may lead to model collapse on this balanced dataset. Because of this behavior, the balanced_accuracy metric is required to maintain a functional decision boundary for the final results.


Milestone 3

Milestone 3 extends the face verification system into a deployable inference pipeline. The system uses an embedding-based representation of face images to compute similarity between two inputs and produce a verification decision. A command-line interface is provided for pair-level inference, returning the similarity score, predicted label, confidence, and latency. The system is designed to run either locally or through Docker using a lightweight inference environment.
The system uses InceptionResnetV1 (pretrained on VGGFace2) from facenet-pytorch to generate fixed-length embeddings for each face image. Images are deterministically loaded from the LFW dataset using tensorflow_datasets, resized, normalized, and then passed into the model to produce embeddings.

The operating threshold used by the inference pipeline is 0.30612244897959173.
Cosine similarity is used to compare embeddings, and higher values indicate the same identity.
Confidence is defined as the absolute difference between the similarity score and the threshold. Predictions farther from the threshold indicate higher confidence, while scores near the threshold indicate lower confidence.

Commands
Run Local Inference:
python run_inference.py --config configs/m1.yaml --left <path_to_left_image> --right <path_to_right_image>


Build the Docker:
docker build -t face-verifier .

Docker Inference (working example):
docker run --rm -v "${PWD}:/app" face-verifier \
  python scripts/run_inference.py --config configs/m1.yaml \
  --left lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg \
  --right lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg

Docker Inference (general code):
docker run --rm -v "${PWD}:/app" face-verifier \
  python scripts/run_inference.py --config configs/m1.yaml \
  --left lfw/<person_name>/<image_1>.jpg \
  --right lfw/<person_name>/<image_2>.jpg

**NOTE:** Image paths must be provided relative to the LFW dataset root using the format:
lfw/<person_name>/<image_file>.jpg


Milestone 3 uses a lightweight dependency file (requirements-m3.txt) to support inference and reduce build complexity. This environment includes only the packages required for embedding generation and inference, while the full project dependencies from Milestones 1 and 2 remain in requirements.txt.


Milestone 4

Milestone 4 is about finishing the system by adding documentation, profiling, and making sure everything can be reproduced from a clean clone. At this point, the goal is not to change the model, but to show how the system works, where it performs well, and where it may fail.

System Overview
The final system uses an embedding-based system for face verification. Images are loaded from the LFW dataset, resized and normalized, and then passed through InceptionResnetV1 (pretrained on VGGFace2) to create embeddings. Cosine similarity is used to compare the embeddings, and a fixed threshold of 0.30612244897959173 is used to decide whether two images belong to the same person. Confidence is defined as how far the similarity score is from the threshold, so predictions further away from the threshold are considered more confident.

System Card
The System Card goes into more detail about the system, including how it is intended to be used, its limitations, common failure cases, and potential fairness risks. It also documents the threshold and assumptions used by the system. It is located at:

reports/System_Card.pdf

Profiling Report
The profiling report shows how the system performs when it runs. It shows preprocessing time, embedding time, scoring time, and total latency. It also looks at how performance changes with different batch sizes. The report is located at:

reports/Profiling report.pdf

How to Run
To reproduce the profiling results, run:

python scripts/profiling.py --config configs/m1.yaml --pairs outputs/pairs/val.csv --batch_sizes 1 5 10 25

This will output latency breakdowns for each stage of the pipeline.

Outputs
The outputs for this milestone include the System Card, profiling report, and reproducibility checklist. These are all stored in the reports/ folder.

Reproducibility Notes
A reproducibility checklist is included so the system can be run from a clean clone step-by-step. It includes setup, ingestion, pair generation, inference, and profiling commands. It is located at:

reports/reproducibility_checklist.md

Final Tag
The final version of the project is tagged as v1.0-final.