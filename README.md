# Detect AI Generated Text

Project repository for Machine Learning Operation 02476

- Aron D. Jacobsen (s194262)
- Daniel Mathias Holmelund (s194245)
- Jacob Sebastian Engel Ketmig (s194264)
- Jens Dieter Kjær Modvig (s173955)
- Pavlos Mylis Adamakis (s222968)


## Project description

The purpose of the project is to familiarize ourselves with the machine learning life cycle, ie. design, model development, and operations, with a focus on the latter stage and using good practices with the tools presented in the course "Machine Learning Operations - 02476", ultimately delivering more robust and reliable solutions to end-users.

In recent years, large language models (LLMs) have become increasingly sophisticated, capable of generating text that is difficult to distinguish from human-written text. Therefore The Learning Agency Lab has set up a competition that challenges participants to develop a machine learning model that can accurately detect whether a student or an LLM wrote an essay. The competition dataset comprises a mix of student-written essays and essays generated by various LLMs.
So the sub-goal of this project is to Detect AI Generated Text and identify which essay is written by a large language model and which by a student.

The model chosen for this project is a lightweight version of BERT (Bidirectional Encoder Representations from Transformers), originally published by Google AI Language, called DistilBERT, and is a part of the Transformer framework built by the Huggingface group.

We opt for this model as it stands out as a potent tool with state-of-the-art performance on numerous NLP tasks, despite its simplicity. Additionally, its compatibility with PyTorch aligns seamlessly with the course curriculum and goal; to use machine learning operation tools - not designing cool AI models. There is a Trainer API in the Transformers library, which allows for easy logging, gradient accumulation, mixed precision, and some evaluations for the training.

The initial dataset consists of approximately 10.000 essays, which are written by students and generated by a mixture of large language models. All essays are written as a response to seven different prompts. More information about the dataset of Kaggle compitition can be found here: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data


## Deployment
To perform inference using the model, please visit:

https://mlopsdetectaigeneratedtext.streamlit.app/


---

## From zero to hero using docker
```
pip install -r requirements.txt
dvc pull
docker compose up
```

## Old Setup


You can 1) run a precompiled conda environment (hopefully on cuda), or 2) perform the manual installation.

### Environment
#### Precompiled environment with Conda

```bash
TODO
```

#### Manual environment installation

Assuming you already have conda installed, run the following:

```bash
conda create -n mlops python=3.11
```

Activate the environment:

```bash
conda activate mlops
```

And install the project dependencies:
```bash
pip install -r requirements.txt
```


Finally install Pytorch based on you computer compatibilities following this guide: https://pytorch.org/get-started/locally/


*And can always find other compatible pytorch modules here: https://pytorch.org/get-started/previous-versions/*

#### Developer

If you want to make changes to the code and immediately see the effect, install the project in editable mode:

```bash
pip install -e .
```

### Download the data

*Request permission:* https://drive.google.com/drive/folders/1apqcOMgmfkuDp4VGnCcmN6Bwx3dE1GC-

Then run:
```bash
dvc pull
```

### Initialize wandb

Login in to wandb:

```bash
wandb login
```

And paste the token from the website.


## Run the scripts


### Training



#### Simple run

Just run the script:

```bash
python ml_ops_detect_ai_generated_text/train_model.py
```

#### Run an experiment

Follow this guide:
1. Note that:
      1. Experiments are configured in the `configs/` folder
      2. Possible hyperparameters are defined in the `configs/models/` and `configs/training/`.
2. You can define an experiments by:
      1. Overwrite the possible hyperparameters in a `configs/experiment/*.yaml` file
      2. Define this file in the `configs/config.yaml` file as: 
           ```yaml
           defaults:
             - experiment: *.yaml
           ```
3. Then run: ```python ml_ops_detect_ai_generated_text/train_model.py```


### Sweep

TODO:

eg.?
```
wandb sweep --project ml_ops_detect_ai_generated_text "./config/sweep/lr_sweep.yaml" 
```

---

## GCP setup

### Activate it

1. Login: `gcloud auth login`
2. CLI: `gcloud auth application-default login`
3. Set project: 
      1. Available projects: `gcloud projects list`
      2. Select a project and run: `gcloud config set project <project-id>`
4. Install dependency: `pip install --upgrade google-api-python-client`

### Data bucket

1. Create a data bucket, is it visible: `gsutil ls`
2. Assuming configured DVC, reconfigure it to: 
      ```bash
      dvc remote add -d remote_storage <output-from-gsutils>
      dvc remote modify remote_storage version_aware true
      ```
3. Push it: `dvc push -r remote_storage`
      1. Failing? try `dvc add data` and follow error specifications

### Train a model

1. Make sure the docker file is working locally: 
      ```bash
      # build image
      docker build -t trainer:latest -f dockerfiles/train_model.dockerfile .
      # run container
      # we set the wandb api key as an environment variable
      docker run -e WANDB_API_KEY=<your_api_key> --name trainer-container -d trainer:latest
      ```
      1. *You can find the API key under your wandb user settings*
2. Push to GCP:
      ```bash
      docker tag gcp_vm_tester gcr.io/<project-id>/trainer
      docker push gcr.io/<project-id>/trainer
      ```
      1. Confirm by going to the container registry
3. Run the image:
      1. Create a Google Compute Engine (GCE)
      2. SSH into that GCE (Google Compute Engine)
      3. Run the docker container:
            ```bash
            docker run -e WANDB_API_KEY=<your_api_key> --name trainer-container -d gcr.io/<project-id>/trainer:latest training.model_path=gs://<bucket-name>/models/ 
            ```

docker run -e WANDB_API_KEY=dd1f2bbf51b8f93e069c89e9798703b40430999b --name trainer-container -d gcr.io/dtumlops-410913/trainer:latest training.model_path=gs://mlops_model_unique/models/ 

docker run -e WANDB_API_KEY=dd1f2bbf51b8f93e069c89e9798703b40430999b --name trainer-container -d trainer:latest

docker tag trainer gcr.io/dtumlops-410913/trainer

docker push gcr.io/dtumlops-410913/trainer

docker run -e WANDB_API_KEY=dd1f2bbf51b8f93e069c89e9798703b40430999b --name trainer-container -d trainer:latest


---

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── project_name  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   ├── dataloaders.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

---

*NOTE:* The check-list is in `reports/readme.md`