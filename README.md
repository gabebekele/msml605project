git clone <repo>
cd repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/ingest.py --config configs/m1.yaml
python scripts/make_pairs.py --config configs/m1.yaml
python -m scripts.bench_similarity