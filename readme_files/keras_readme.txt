Application Notes: 
+ Layers (and the underlying datastructures) are created when when call the layer functions. When models use these layers, the data structures are the same for all models using those base layers. So, as we create different model names for hose base layers with Model(), we are just creating different ways to reference those base layers.
+ Use PIL insead of cv2 since keras is dependent on PIL. This makes sure that the formatting is consistent with keras functions. 
+ Configuration parameters: We can get any configuration parameters for model and layer class with model.name, etc.
+ VGG assumes a BGR format. However, Keras uses PIL to initially load images, and thus often used a RGB format. Be careful with going back and forth between thes formats.  



Programming Notes: 
+ model.summary(): Provides a summary of the model. 
+ model.input/model.output: Get the input and output layer of a model. Output is the final layer. Input is usually the input layer. 
+ model.get_config() or layer.get_config(): Gets the setup configuration of the model or layers. 
+ model.plot_model(): Plots the model, and outputs as image. 
+ Generators in Keras: flow_from_directory and flow() return generator objects. Generator objects are iterable objects that generate a result when we call next() on these objects. Generators can either be 1) a function that yields results every time next() is called on the function or 2) an object that is iterable through next, and generates results in real time. In Keras, we can use either approach (with flow_from_directory using the object approach). Generators allow for real-time data loading, instead of preloading the data. 