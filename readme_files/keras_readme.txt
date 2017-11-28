Application Notes: 
+ Layers (and the underlying datastructures) are created when when call the layer functions. When models use these layers, the data structures are the same for all models using those base layers. So, as we create different model names for hose base layers with Model(), we are just creating different ways to reference those base layers.
+ Use PIL insead of cv2 since keras is dependent on PIL. This makes sure that the formatting is consistent with keras functions. 
+ Configuration parameters: We can get any configuration parameters for model and layer class with model.name, etc. 



Programming Notes: 
+ model.summary(): Provides a summary of the model. 
+ model.input/model.output: Get the input and output layer of a model. Output is the final layer. Input is usually the input layer. 
+ model.get_config() or layer.get_config(): Gets the setup configuration of the model or layers. 
+ model.plot_model(): Plots the model, and outputs as image. 