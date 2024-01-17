---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [x] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [x] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

3

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s194245, s194262, s194264, s173955, s222968

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:


For the machine learning aspect of the project, Pytorch was utilized to define the model architecture and implement the dataloaders. To reduce the amount of boilerplate code, the high-level Pytorch Lightning framework was employed, simplifying the training process by using the Trainer object. Huggingface was used to import a pretrained DistilBert model, which greatly simplified the model implementation.

Hydra was employed to ensure reproducibility. This approach separated the definition of hyperparameters from the model itself, making version control of the hyperparameters much easier. Additionally, Weights and Biases was used as a logging tool, providing insights into each training run, such as reporting losses.

Docker played a crucial role in the deployment phase. Building the Docker image in the cloud ensured system-level reproducibility of the model pipeline. 

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

Miniconda was used to manage the depenencies for this project. To automatically generate the list of dependencies ```pip freeze > requirements.txt``` can be used to extract all install dependencies from a environment. However, in to limit the size of the requirements file it was manually updated.

Doing the following will get a new team member started with the project:

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
pip install -e .
```

Finally install Pytorch based on you computer compatibilities following this guide: https://pytorch.org/get-started/locally/

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:
> 
In our project, we adopted the cookiecutter data science template as the foundational structure. Key files such as predict_model.py and train_model.py were developed, along with additional utility files, for instance, ml_ops_detect_ai_generated_text /data/init.py, which is responsible for dataset construction. The make_dataset.py file was left unfilled since our data necessitated minimal preprocessing, which we opted to perform during runtime instead
> We have included a folder named 'tests', which contains our unit tests related to the data and the model. Additionally, a '.dvc' folder has been created to hold metadata files associated with data versioning, utilizing cloud services.



### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

Ruff was utilized as both a linter and a formatter using GitHub actions to ensure PEP8 compliancy. By implementing it this way these checks are made automatically when pushing to main. In our case multiple redundant import statements were removed by this check. 

In larger project with a big team collaborating, ensure a cohesive formatting helps with code readability and alleviates compatibility issues. 


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- question 7 fill here ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- question 8 fill here ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

During our project, we used the technique of involving both branches and pull requests. Whenever a new feature was under development or testing, we initiated a separate branch for it. Upon completion of the feature, we would create a pull request to merge this feature branch back into the main branch.Incase any of any uncertainty we had a branch for concerning the replacement of the model that a malfucntion could produce.As a standard practice, we ensured that the pull requests were always reviewed and approved by a different member of the group. This approach was used to make sure that theres collaboration and to maintain a review process.


### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:
> User

Initially, we utilized Google Drive for data storage, with DVC (Data Version Control) handling the management. However, to enhance our storage solution, we later tried to experiment with our data from Google Drive to Google Cloud's Data Storage buckets, continuing to manage it with DVC, but the outcome was the same.Both options enabled us to modify data as necessary while preserving its historical record, whie they effectively addressed the challenges of storing large data sets in a git repository.To conclude in both version the approach to ensured uniformity in data access and structure across all team members, which benefits for streamlining our pipelines and deployment processes.



### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:
The hyperparameter values were defined in hydra configuration files, and we integrated a hydra decorator into the 'train' function of the train_model.py file. This integration facilitated the insertion of hyperparameters and enabled the logging of both experiment performance and the hyperparameters utilized. To initiate an experiment.To run the function, just write in the terminal 'python ml_ops_detect_ai_generated_text/train_model.py' directly would produce the outcome.


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Configuration files play a crucial role in our setup. These files help in clearly linking hyperparameters with the corresponding model performance, leaving no ambiguity. Additionally, DVC is implemented to ensure data consistency across different machines. Furthermore, the use of Docker facilitates the creation of a stable environment, maintaining specific library versions, dependencies, and packages. This setup is vital to prevent any variations in library behaviors from impacting our results.We couldn't achieve complete reproducibility was in random processes, such as the shuffling of batches, since we didn't regulate these with seeds. Moreover, it's important to mention that even when seeds are used, PyTorch doesn't ensure the outputs will be exactly the same every time.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:
![image](https://github.com/DanielHolmelund/ml_ops_detect_ai_generated_text/assets/114672733/43d1aec0-9f3f-45f7-ab2e-58c8cf92aa7d)

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

The trainer.dockerfile defines an image that we utilized for local model training, using data we already had. This process saved the trained model's state in a shared folder.

On the other hand, predict.dockerfile's image was responsible for loading this saved model state, running tests on some pre-existing data, and then calculating the model's accuracy. These two dockerfiles, used for local deployment, mainly differ in their initial commands: one initiates the training process while the other focuses on evaluation.

For deployment on Google Cloud Platform (GCP), we used the dockerfile located in api/gcp_run/dockerfile. Unlike the others, this one doesn’t specify an entry point; it uses the CMD command to launch the inference API.


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- question 16 fill here ---

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

On the Google Cloud Platform, we utilized a variety of services. For remote storage of our data and Docker images, we relied on Cloud Storage. Additionally, we established a GCP Bucket, designated as the remote repository for our data version control, allowing us to efficiently push to and pull data from the cloud.

We configured triggers to automate the building of trainer and predictor Docker images. These triggers were activated by any push to the main branch of our cloud repository, which mirrors our git repository. All the container images created throughout the project were stored and are accessible in the Container Registry.

For building and deploying the inference API, we employed Cloud Run and Cloud Functions, leveraging their capabilities for our project's needs.


### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:
 Vertex AI in google cloud,is meant to be used for training the model in the cloud. 
--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

In general, this subject wasn't given top priority in monitoring because Google Cloud appears to be quite comprehensive in its monitoring capabilities, covering aspects like security, user activity, and performance. Moreover, there's an option to establish a maximum on the number of requests.
Additionally, "Cloud Armor Network Security" offers safeguards against threats such as DDoS attacks. Due to limited time..
--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

Configuring our deployment setup and getting everything up and running on Google Cloud turned out to be the toughest and most time-consuming task for us. A lot of this difficulty came from our team's relative unfamiliarity with Google Cloud's machine learning features – even just getting the hang of the interface was quite a task. We had to rely heavily on a mix of trial and error, along with a lot of detailed reading of tutorials and guides, to get our whole pipeline operational. It was a bit of a journey, but in the end, we managed to get everything working as intended.
--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:
In general students contributed equally to the project, and we often worked side-by-side. Everyone worked on the documentation of the project as well.

In case specific contributions are needed:

s194245
s194262
s194264
Pavlos Adamakis-s222968
s173955 

--- question 27 fill here ---
