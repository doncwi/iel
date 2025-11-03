# iel
image embedding lab

BACKBONE:
To build this MVP fastembed was used and heavily relied on with the "Qdrant/clip-ViT-B-32-vision" model being used for embedings

ASSUMPTIONS:
- images used in training dont have cluttered backgrounds
- images used in training dont have low entropy
- images are not corrupted
- images are not blank
- speed was a factor
- download/setup times should not take several GB or > 15 minutes

HOW TO RUN:

1. Install libs using
pip install fastembed pillow numpy scikit-learn tqdm

2. To Run embeds (do first before running search/analyze): 
python embedlab.py embed --images-dir ./assets/images --out ./index

3a. To run search:
NOTE: this code supports search batching, include more than 1 file in queries and the search will be auto batched
python embedlab.py search --index ./index --query-dir ./assets/queries --k 5 --json

3b. To run Analyze:
python embedlab.py analyze --index ./index --dup-threshold 0.92 --anomaly-top 8 --json


References/citations:
https://github.com/qdrant/fastembed
https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset