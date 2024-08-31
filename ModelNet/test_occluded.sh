# testing multi-level PointMLS and save their tesing score
python test.py --model PointMLS_basic
python test.py --model PointMLS_512
python test.py --model PointMLS_256
python test.py --model PointMLS_128
# ensemble their testing score, you can change the ensemble weights by editing the "weights" parameter in ensemble.py
python ensemble.py