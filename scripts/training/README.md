## Training

Under the `./scripts/training/` directory, an example training script is provided for reference with arguments.

```console
$ python train.py --model --level [-h]

Description:
  Train an ML/DL model

Required arguments:
  --model        Choice of model to train: lr, dt, nn, linearsvc, nusvc, knn, all (default: all)
  --level        Which level of information (1-5) to use, as described in the methodology (default: 1)

Help dialog:
  -h, --help     Show this help message and exit
```

The training script has verbose logs being stored in the form of `*.txt` files within the `Inferences` directory with the date and model information, pickled/converted model file, and evaluation performance (ROC curve/training loss-accuracy graphs).
