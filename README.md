# HANet_v2

Here is the work base on [HANet](https://github.com/austin2408/HANet) but change the last layer to 4 class (4 angles : 90, -45, 0, 45), so this version can directly output the complete orientation of the object (the old version need to do 8 times predictions to find the object angle) .<br>

![Teaser](figure/pred_sample.png)

HANet (backbone : ResNet101)<br>
![Teaser](figure/model_structure.png)

# Dataset Overview
![Dataset](figure/datasets.png)
