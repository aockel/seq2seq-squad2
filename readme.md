# Udacity project Chatbot with seq2seq architecture using SQuAD2
This project was part of the [Udacity Deep Learning Nano Degree Program](https://www.udacity.com/course/deep-learning-nanodegree--nd101)
that was teaching basics on building seq2seq architectures using LSTM cells.

In this project, we did build and train a custom seq2seq architecture on the SQuAD2 dataset,
leveraging the different skills learned during the course mentioned above.

## Model Architecture
### LSTM with Seq2Seq
In this notebook, I've chosen to apply the seq2seq architecture to build a model using LSTM layers in the encoder and decoder.

## Data Set
You’ll be training the model on the SQuAD2 dataset which is available within the torchtext package (see `requirements.txt`).

# Getting Started
So, our goal is to create a model that can generate answers based on user input questions.
We’ll go through the following steps:
1. Load, prepare and clean data
2. Define encoder, decoder and seq2seq classes
3. Train the model network
4. Visualize the loss over time and some sample, generated images
5. Provide an interactive chatbot via the console

## Requirements
It is recommended to train the model on GPU.

### Initial setup within notebook
Install requirements by adding a cell on the top and run
```bash
!pip install -r requirements.txt
```

### Initial set-up of pyenv virtualenv in case of a local dev env
Install python base version defined in `.python-base-version` file.
```shell
pyenv install --skip-existing $(cat .python-base-version) 
```
Create virtualenv with a name defined in `.python-version` using the installed python base version.
```shell
pyenv virtualenv $(cat .python-base-version) $(cat .python-version)
```

Add the following to your .zshrc file so that the terminal will pick up and activate your virtual env 
automatically as soon as you enter a folder that contains a `.python-version` file. 
```shell
# set root directory for pyenv
PYENV_ROOT=~/.pyenv
# Alias
alias brew='env PATH="${PATH//$(pyenv root)\/shims:/}" brew'

# define plugin
plugins=(virtualenv)
POWERLEVEL9K_RIGHT_PROMPT_ELEMENTS=(status virtualenv)

export PATH=$PYENV_ROOT/shims:/usr/local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
eval "$(pyenv init -)"
source /usr/local/opt/chruby/share/chruby/chruby.sh
source /usr/local/opt/chruby/share/chruby/auto.sh
chruby ruby-3.1.1
```

### Install python requirements
```shell
pip install -r requirements.txt
```