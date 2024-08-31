# training PointMLS in a multi-level manner (1024, 512, 256, 128)
python main.py --model PointMLS_basic --epoch 65
python main.py --model PointMLS_512 --epoch 65
python main.py --model PointMLS_256 --epoch 65
python main.py --model PointMLS_128 --epoch 65