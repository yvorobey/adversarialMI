## About
The code repository for paper 
"Anatomical Context Protects Against Adversarial Perturbations on Medical Imaging Deep Learning"

## Run codes
To run the experiments, some parameters are needed.

The **first parameter** is the attack method we want to run. 
In this repository, four attacks are implemented.

* gsm_exp: the <img src="https://latex.codecogs.com/gif.latex?l_\infty" title="l_\infty" />
attack for a single image
* l2_exp: the <img src="https://latex.codecogs.com/gif.latex?l_2" title="l_2" /> 
attack for a single image
* l0_exp: the <img src="https://latex.codecogs.com/gif.latex?l_0" title="l_0" />
attack for a single image
* multi_gsm: the <img src="https://latex.codecogs.com/gif.latex?l_\infty" title="l_\infty" />
attack for multiple images
    
The **second parameter** is whether the model we want to attack used the MAS features.
If the value with-mas is True, the model use the image data with the MAS
features to predict age. Otherwise no features would be used.

The **third parameter** is used to choose whether we want to maximize the 
prediction age or minimize the prediction age by adding the perturbation.
The value can be chosen to be max or min.

We can also choose the instances to attack by specifying the range 
[start, start+instances] and whether a sample perturbation would be saved
by specifying the value of save-data.

As an examples, to maximize the prediction age in the model with MAS features
using l2 attack, we can do

    python main.py --mode=l2_exp --with-mas=true --direction=max --start=0 --instances=1 --save-data=true

The code works on Python 3.5.3.