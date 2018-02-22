# TASK 4, COMP 551:: Adversarial Machine Learning
All source code are put under *src* subdirectory.

### Project report file
adversarial-attacks-image.pdf

### Link to the video
[Link](https://www.youtube.com/watch?v=f_g79JxyPjI)

### Tools used, and other dependencies
1. [Cleverhans, OpenAI](https://github.com/openai/cleverhans)
2. Tensorflow, version 3.4 and above
3. Keras

### Generated adversarial examples
https://drive.google.com/drive/folders/0B2pDcGINKNIOVmdBc1Z4VlBGVGs?usp=sharing

### Attack Files
attack_fgsm.py  
attack_jsma.py  
attack_blackbox.py  

### Defense Files
def_adv.py

#### Defensive Distillation
*simple_NN_laddened_with_defensive_distillation.py*  
	Contains the implementation of a protection of a network with defensive distillation.  
	Toggle comment between lines 74 and 75 to see performance on legitimate and adversarial examples.
*simple_NN_with_no_defensive_distillation.py*  
	Contains a model with no protection from any threat.  
	Toggle comment between lines 50 and 51 to see performance on legitimate and adversarial examples.

### Data Visualization and Results Generation Files
visualize.py  
images.pysimple_NN_with_no_defensive_distillation.py  

### Comparison Between Different Methods
results.py

### Other directions
1) All the files can be run using "python <filename>"
2) Depending on the location of datafile, the path has to be edited inside the file.  
Due to size limitations, adversarial example files are uploaded on google drive. Link has been shared at the top of this file.
