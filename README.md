## mlp-backpropagation
Backpropagation -- The "2-Class" Problem
#Introduction  

For this project we define a simple classification problem based on the position of (x,y) points in a square.  
![Figure 1](http://preet.sesolution.com/CSC5542/p4/img/intro1.jpg)

Figure 1: Points in the square are labeled as either Type A (inside the circle) or Type B (outside the circle).  
To follow Lippmann's parameters: square with side of length 4, circle centered at the origin with radius 1. The goal of the project is train a neural net to learn the difference between Type A and Type B points.  

#Architecture:  
Figure 2 shows the recommended architecture for this problem.  Notice that there are:  
 
1.   Two input units, one for the x - coordinate and one for the y - coordinate;  
2.   Two output units, on for classifying the input as type A, and one for type B;  
3.   Eight hidden units.  

![Figure 2] (http://preet.sesolution.com/CSC5542/p4/network.jpg)  

Figure 2: Architecture for the two-class backpropagation example. Note: the hidden and output layer neuron biases are not shown, but they are always there  

Training: First the weights are randomly initialized as small positive and negative values.  
Then points of known type (A or B) or randomly chosen from the square, their (x,y) values fed forward through the network and the output value for the two output neurons is computed.  The desired output for type A points is (1,0) and the desired output for type B points is (0,1).  The error between the desired and the actual outputs are then used to update the weights using the error backpropagation method (see class notes for the specific steps in the backprop method).
 
Training Set: This example is a little unusual in that there is no "fixed" training set.  Of course, you could pick, say, 1000 points of type A and 1000 of type B, shuffle them up, save them, and call that the "training set".  Then, we have a situation that looks more like the typical backprop application (i.e.: given a set of known samples of each type).  For training the network you keep cycling through the training set ("epochs") -- keep cycling until the classification error is very small.  Then to see if the network generalizes correctly, you show it some points that were not in the training set (another set of, say, 100 from type A and 100 from type B).
 
One student (see Michael Parry's project in the Class Notes) decided to pick only points that were very close to the boundary of the circle.  This sped up the training process. Although this is very instructive, it is "cheating" since it uses knowledge of where the boundary is in this case --- that's exactly what the neural net is supposed to compute for us, so, if we knew that they we would not need the net.  But, it shows us that the points near the boundary carry a lot of "weight" in the training process.
 
For this particular case, however, it is reasonable to train the network by picking random points, computing their classification (A or B) on the spot (in other words, compute the distance from the origin and label them "A" or "B"), and using that to compute the output error, and applying the backpropagation method etc.  This way there is no need to specify an explicit set of points ahead of time.
 
Testing:  After training the network, it should be tested to see if it is getting the classifications correct.  Feed in another 100 randomly chosen points and compute the % error (the fraction of the cases were incorrectly classified).
