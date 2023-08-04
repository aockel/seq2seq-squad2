## initial set-up of pyenv virtualenv
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

## install python requirements
```shell
pip install -r requirements.txt
```