## Training

Under the `./scripts/training/` directory, an example training script is provided for reference with arguments.

```
python scripts/create_hf_dataset.py \
    -d /PATH/TO/mmnlu-eval/data \
    -o /PATH/TO/hf-mmnlu-eval \
    --intent-map /PATH/TO/massive_1.0_hf_format/massive_1.0.intents \
    --slot-map /PATH/TO/massive_1.0_hf_format/massive_1.0.slots
```

### scikit-learn models (ML)

### TensorFlow models (DL)
