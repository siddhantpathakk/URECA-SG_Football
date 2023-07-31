# Creating an Expected Goal Model for Singapore football

## Installation

To ensure a clean and isolated installation, it is recommended to use a virtual environment. Conda is a popular choice for managing virtual environments, providing a seamless experience for installing dependencies.

### Using Conda (Recommended)

1. If you have Conda installed, navigate to the project's root directory (where the `ureca.yml` file is located) using the terminal or command prompt.
2. Create a new Conda environment using the `ureca.yml` file:
   ```
   conda env create -f scripts/ureca.yml
   ```
3. Activate the newly created Conda environment:
   ```
   conda activate ureca
   ```

If you use pip/venv, please install the dependencies using the `requirements.txt` file:

```
pip install -r scripts/requirements.txt
```

Once the installation is complete, you can start using the code and running the scripts within the virtual environment. Remember to activate the environment each time you work on the project.

> Note: Using Conda is recommended as it simplifies the installation process and ensures a consistent environment. If you already have Conda installed, it is advised to create the environment using the `ureca.yml` file directly, as it contains the necessary dependencies for the project.

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
