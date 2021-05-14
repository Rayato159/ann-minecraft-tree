# Minecraft_Tree
Creator: Ruangyot Nanchiang 05/13/2021  
Lecturer: Dr.Suradet Tantrairatn  

This project is to study about multilayer perceptron.

## Create Datasets
My inspriation is from "Minecraft". Then I pick types of tree to create datasets because they are random shape for each and easy to create a picture of tree by coding without drawing.  

Two types of tree I pick
1. [Oak](https://minecraft.fandom.com/wiki/Oak)  
2. [Spruce](https://minecraft.fandom.com/wiki/Spruce)  

I generate all images of tree by drawing and try some random by each types pattern with OpenCV and saved all images in ".npy" files.  
Left is "Oak" and Right is "Spruce". Sample here.  
![Sample](https://github.com/Rayato159/Minecraft_Tree/blob/main/sample.png)

## Training Model
As I mention before, I used MLPClassification to train a model. 
And I create a class "ModelTester" for easy to test all activations and solvers.  
```python
class ModelTester:
    def __init__(self, df, activation, solver):
    
        self.activation = activation
        self.solver = solver
        self.df = df
        
        self.clf = MLPClassifier(activation=self.activation, solver=self.solver, random_state=1, max_iter=1000)
        ...
        
        print(f"activation: {self.activation} \tscore: {self.score}")
```
Thus I can pass the activation and solver by just create a object with this class.

## Result
Accuracy score of this model = 1.0 (Score from test data).
