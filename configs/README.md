This folder stores all configuration variables used for launching training runs (and evaluating the results from those runs) in the form on config files.

The ```main.py``` script has a ```--config``` argument which can be the path to any of the "config.*.json" files in this folder. Of course you can also write your own config file.
Thus one "config.*.json" files points to all parameters used for the run.

As we perform quite a lot of different experiments requiring only a few parameters to change, I have setup a hierarchical system of config files allowing default values that can be overwritten.
This way a single parameter can be shared by all experiments and can be changed for all of them by modifying it in only one location.
A config is stored as a JSON dictionary. This config dictionary can store nested dictionaries of parameters. 
Whenever the "defaults_filepath" key is used in a config file, 
its value is assumed to be the path to another config file whose dictionary is loaded and merged with the dictionary that had the "defaults_filepath" key.
The keys alongside the "defaults_filepath" key specify parameters that should overwrite the default values loaded from the "defaults_filepath" config file.

To illustrate how this works, let's say the config folder looks like this:
```
config 
|-- config.defaults.json
`-- config.my_exp_1.json
`-- config.my_exp_2.json
```
 
Let's say config.defaults.json is:
```json
{
  "learning_rate": 0.1,
  "batch_size": 16
}
```  

And config.my_exp_1.json is:
```json
{
  "defaults_filepath": "configs/config.defaults.json"
}
```

And config.my_exp_2.json is:
```json
{
  "defaults_filepath": "configs/config.defaults.json",
  
  "learning_rate": 0.01
  
}
```

When loaded by the ```main.py``` script, they will be expanded into the following.

config.my_exp_1.json:
```json
{
  "learning_rate": 0.1,
  "batch_size": 16
}
```

config.my_exp_2.json:
```json
{
  "learning_rate": 0.01,
  "batch_size": 16
  
}
```

When a lot of parameters are used by the actual config files we used, it is thus very easy to know that all "my_exp_2" does is change the learning rate.
Also if we want to change the batch size for all experiments, all we have to do is change its value in "config.defaults.json".
This principle of using the "defaults_filepath" key to point to another config file can be used in nested dictionary parameters as well.
A config file is thus the root of a config tree loaded recursively.