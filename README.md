The advent of deep learning has revolutionized medical image analysis, offering promising 
avenues for early disease detection and diagnosis. This paper presents a novel approach utilizing 
DenseNet models for the detection of leukemia, a critical hematologic malignancy with diverse 
manifestations across different patient populations. 
DenseNet, characterized by its dense connections between layers, has demonstrated superior 
performance in various image classification tasks by enhancing feature propagation and 
encouraging feature reuse. Leveraging this architecture, we propose a robust framework for 
automatic leukemia detection from peripheral blood smear images. 

 RandomSearch: - 
"Random search" is a simple yet effective algorithm used in machine learning and deep learning 
for hyperparameter optimization. Hyperparameters are settings that are external to the model itself 
and affect its performance, such as learning rate, batch size, and the number of layers in a neural 
network. 
 Define a Search Space: Specify the range or distribution for each hyperparameter that you want 
to tune. For example, you might specify that the learning rate should be sampled uniformly from 
the range [0.001, 0.1], the batch size should be sampled from [32, 64, 128], and so on. 
 Set the Budget: Decide how many random combinations of hyperparameters you want to try. This 
budget can be based on computational resources or time constraints. 
 Train and Evaluate: For each sampled configuration, train the model on the training data using 
those hyperparameters and evaluate its performance on a validation set. 
 Select the Best Configuration: Keep track of the performance of each configuration based on 
some evaluation metric (e.g., accuracy, loss, etc.). 
 Repeat: Repeat steps 3-5 until the budget is exhausted. 
 Final Model Training: After all configurations have been evaluated, select the configuration that 
performed the best on the validation set. Optionally, you can retrain the model using this 
configuration on the entire training dataset.
