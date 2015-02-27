# Handwritten Digits Recognition, Neural Network    
####Experiment with handwritten digits recognition, with Octave neural network "nnet" package (or MATLAB Neural Network toolbox).    

Based on the Prof. Andrew Ng (Stanford)'s Machine Learning course.   

Explanation note in Blogger: [this link](http://blog.wijono.org/2015/02/handwritten-digits-recognition.html) or [redirect link](http://plus8888.blogspot.com/2015/02/handwritten-digits-recognition.html).      

**Note:** This is NOT solution to the course's assignment. This is merely my personal experiment to the same problem, in smaller scale, by utilizing Octave package / MATLAB toolbox.    
        
Link to Octave [nnet](http://octave.sourceforge.net/nnet/index.html) package.     
    
#####For better accuracy,     
1. increase `hidden_neurons` up to 25 nodes (too many nodes may result in overfitting),    
2. increase sample size `train_size` up to 5000,        
    
Please remember, total amount of sample data is 5000. While default value for test and validation data is started from 4000 and 4800 in the code. So if we want `train_size` to be larger than 200, than we need to change `ntest` and `nvali` accordingly, so that they won't exceed the size.   
    
In the code we use randomized initial weights for symmetry breaking. Please note that Prof Ng. course's code uses trained weights.    
    
For newer MATLAB, warning maybe thrown to complain `newff()` as it uses obsolete format. In this case we need to refer to the latest documentation of the Neural Network Toolbox.    
    
ANN's cost function is not convex / concave. So the converged cost solution in ANN may not refer to global minimum, there may be some local minima. The accuracy could vary.     
    
#####Result:     
This implementation need much larger amount of memory allocation, and much slower execution time (training), compared to if we develop the code from scratch (as in the Prof. Ng's course).   
    
- Training set accuracy is around 63% with the small sample (120) and small hidden neurons (16).    
- The "developed from scratch" course's code (with backpropagation) yields 74% accuracy, with the same small sample (120) and small hidden neurons (16).            
- While, larger sample (5000) and more hidden neurons (25) yields 95% for 50 epochs, and 99% for 500 epochs.       
    
--------------------------------------    
