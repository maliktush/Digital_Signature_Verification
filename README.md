# Writer Independent Offline Signature Verification using

# Download Datasets
```bash
cd data
bash cedar.sh
bash bhsig260.sh
```

# Data Preperation
Prepare Data and split into test/train sets.

From the root folder, run
```bash
python3 data_preperation.py
```

# Train
```bash
python3 train.py --dataset <choices:['cedar', 'bengali', 'hindi']>  --model <choices:['model1', 'model2', 'model3']> --lr<learning_rate>
```