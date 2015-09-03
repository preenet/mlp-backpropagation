/**
 * @author Pree Thiengburanthum
 * CSC5542 Neural Networks
 * Network.java
 */
import java.util.*;

public class Network {
	private int maxEpochs;
	private int maxTrain;
	private int maxTest;
	private int inputSize = 2;
	private int hiddenSize = 8;
	private int outputSize = 2;
	private int simIter;
	private int success;
	private int failure;
	private int converge;
	private Node[] inputNode;
	private Node[] hiddenNode;
	private Node[] outputNode;
	private double stepSize;
	private double threshold;
	private double weightRange;
	private double momentum;
	private double squareSumError;
	private double sRate;

	private double[] hiddenDelta;
	private double[] outputDelta;
	private double[][] inputWeight;
	private double[][] oldInputWeight;
	private double[][] outputWeight;
	private double[][] oldOutputWeight;
	
	private Vector<Point> myTrainPoints;
	private Vector<Point> myTestPoints;
	private Vector<Double> ssError;
	private StringBuffer successRate;
	private Vector <Double> outAct;

	
	public Network(int maxEpochs, int maxTrain, int maxTest, double stepSize, double threshold,
			double weightRange, double momentum) {
		this.maxEpochs = maxEpochs;
		this.maxTrain = maxTrain;
		this.maxTest = maxTest;
		this.stepSize = stepSize;
		this.threshold = threshold;
		this.weightRange = weightRange;
		this.momentum = momentum;
		this.simIter = 0;
		this.success = 0;
		this.failure = 0;
		this.sRate = 0;
		this.squareSumError = 0;
		
		inputNode = new Node[inputSize];
		outputNode = new Node[outputSize];
		hiddenNode = new Node[hiddenSize];
		hiddenDelta = new double[hiddenSize];
		outputDelta = new double[outputSize];
	
		
		inputWeight = new double[inputSize][hiddenSize];
		oldInputWeight = new double[inputSize][hiddenSize];
		outputWeight = new double[hiddenSize][outputSize];
		oldOutputWeight = new double[hiddenSize][outputSize];

		myTrainPoints = new Vector<Point>();
		myTestPoints = new Vector<Point>();
		ssError = new Vector<Double>();
		
		successRate = new StringBuffer();
		
	}// end constructor
	
	public Vector<Point> getTrainingPoints() {
		return this.myTrainPoints;
	}// end method getTrainingPoints
	
	public Vector<Point> getTestPoints() {
		return this.myTestPoints;
	}// end method getTestPoints
	
	public String getSuccessRates() {
		return this.successRate.toString();
	}// end method getSuccessRate
	
	public double getFinalSuccessRate() {
		return this.sRate;
	}// end method getFinalSuccessRate
	
	public Vector<Double> getSSError() {
		return this.ssError;
	}// end method getSSError
	
	public int getConverged() {
		return this.converge;
	}// end method getConverged
	
	
	public void runSim() {
		boolean flag = false;
		while(simIter < maxEpochs) {
			squareSumError = 0;
			if(sRate == 95 && !flag) {
				converge = simIter;
				flag = true;
			}
			for(int i=0; i<maxTrain; i++) {
				trainNetwork(myTrainPoints.elementAt(i)) ;
			}
			ssError.add(squareSumError / myTrainPoints.size());
			sRate = ((success * 100.0)/maxTrain);
			successRate.append(sRate + "\n");
			success = 0;
			simIter++;
		}
		for(int i=0; i<maxTest; i++)
			testNetwork(myTestPoints.elementAt(i));

	}// end method runSim
	
	public void initializeNetwork() {	
		initializePoints();
		initializeNodes();
		initializeWeights(weightRange);
	}// end method initializeNetwork

	private void trainNetwork(Point p) {
		double[] desiredOutput = new double[outputSize];

		
		inputNode[0].setActivation(p.getX());
		inputNode[1].setActivation(p.getY());
		
		if(p.getType() == 'A') {
			desiredOutput[0] = 1.0;          
            desiredOutput[1] = 0.0;
        }else {
            desiredOutput[0] = 0.0;          
            desiredOutput[1] = 1.0;
        }
		
		feedForward();
		
		squareSumError += backPropagation(desiredOutput);
		
		for(int i=1; i<outputSize; i++) 
			if((Math.abs(desiredOutput[i] - outputNode[i].getActivation()) < threshold))
				success++;
	}// end method trainNetwork
	
	private void testNetwork(Point p) {
		inputNode[0].setActivation(p.getX());
		inputNode[1].setActivation(p.getY());
		
		feedForward();
		if(!(p.getType() == classify(outputNode))) 
			failure++;
		else {
			p.setSuccess();
		}
	}// end method testNetwork
	
	private char classify(Node[] output) {
		if(outputNode[0].getActivation() > 0.5 && outputNode[1].getActivation() < 0.5) 
			return 'A';
		else if(outputNode[0].getActivation() < 0.5 && outputNode[1].getActivation() > 0.5) {
			return 'B';
		}
		return '-';
	}// end classify
	
	private void feedForward() {
		double temp;
		for(int i=0; i<hiddenSize; i++) {
			temp = 0.0;
			for(int j=0; j<inputSize; j++)
				temp += inputNode[j].getActivation() * inputWeight[j][i];
			hiddenNode[i].setActivation(sigmoid(temp + hiddenNode[i].getBias()));
		}
		
		for(int i=0; i<outputSize; i++) {
			temp = 0.0;
			for(int j=0; j<hiddenSize; j++) 
				temp += hiddenNode[j].getActivation() * outputWeight[j][i];
			outputNode[i].setActivation(sigmoid(temp + outputNode[i].getBias()));
		}
		
	}// end method feedForward
	
	private double backPropagation(double[] desiredOutput) {
		double temp;
		double sumError = 0;
		// calculate error
	    for(int i=0; i<outputSize; i++)
            outputDelta[i] = (outputNode[i].getActivation() * (1 - outputNode[i].getActivation())) *
            				 (desiredOutput[i] - outputNode[i].getActivation());
  
	    
	    for(int i=0; i<hiddenSize; i++) {
	    	temp = 0.0;
	    	for(int j=0; j<outputSize; j++) 
	    		temp += outputDelta[j] * outputWeight[i][j];
	    	hiddenDelta[i] = hiddenNode[i].getActivation() * (1.0 - hiddenNode[i].getActivation()) * temp;
	    }
	    
	    
	    // update weights
	    for(int i=0; i<inputSize; i++) {
	    	for(int j=0; j<hiddenSize; j++) {
	    		temp = inputWeight[i][j] + (stepSize * hiddenDelta[j] * inputNode[i].getActivation()) +
	    			   (momentum * (inputWeight[i][j] - oldInputWeight[i][j]));
	    		oldInputWeight[i][j] = inputWeight[i][j];
	    		inputWeight[i][j] = temp;
	    	}
	    }
	    
	    for(int i=0; i<hiddenSize; i++) {
	    	for(int j=0; j<outputSize; j++) {
	    		temp = outputWeight[i][j] + (stepSize * outputDelta[j] * hiddenNode[i].getActivation()) +
	    			   (momentum * (outputWeight[i][j] - oldOutputWeight[i][j]));
	    		oldOutputWeight[i][j] = outputWeight[i][j];
	    		outputWeight[i][j] = temp;
	    	}
	    }
	    
	    // update bias
	    for(int i=0; i<hiddenSize; i++) {
	    	temp = hiddenNode[i].getBias() + (stepSize * hiddenDelta[i]) +
                   (momentum * (hiddenNode[i].getBias() - hiddenNode[i].getOldBias()));
	    	hiddenNode[i].setOldBias(hiddenNode[i].getBias());
	    	hiddenNode[i].setBias(temp);
	    }
	    
	    for(int i=0; i<outputSize; i++) {
	    	temp = outputNode[i].getBias() + (stepSize * outputDelta[i]) +
	    		   (momentum * (outputNode[i].getBias() - outputNode[i].getOldBias()));
	    	outputNode[i].setOldBias(outputNode[i].getBias());
	    	outputNode[i].setBias(temp);
	    }
	    
	    for(int i=0; i<outputSize; i++)
	    	sumError += Math.pow((desiredOutput[i] - outputNode[i].getActivation()), 2.0);
	    return sumError;
	}// end method backPropagation
	

	private double sigmoid(double d) {
		return 1/(1 + Math.exp(-1 * d));
	}// end function sigmoid
	
	private void initializePoints() {
		for(int i=0; i<maxTrain; i++) 
			myTrainPoints.add(new Point());
		
		for(int i=0; i<maxTest; i++)
			myTestPoints.add(new Point());
	}// end method initializePoints
	
	private void initializeNodes() {
		for(int i=0; i<inputSize; i++)
			inputNode[i] = new Node(0, 0);
		
		for(int i=0; i<hiddenSize; i++)
			hiddenNode[i] = new Node(Math.random(), randomBias());
		
		for(int i=0; i<outputSize; i++)
			outputNode[i] = new Node(Math.random(), randomBias());
	}// end method initializeNodes

	private void initializeWeights(double weightRange) {
		for(int i=0; i<inputSize; i++) {
			for(int j=0; j<hiddenSize; j++) {
				inputWeight[i][j] = randomBias() * weightRange ;
				oldInputWeight[i][j] = inputWeight[i][j];
			}
		}
		
		for(int i=0; i<hiddenSize; i++) {
			for(int j=0; j<outputSize; j++) {
				outputWeight[i][j] = randomBias() * weightRange;
				oldOutputWeight[i][j] = outputWeight[i][j];
			}
		}
	}// end method initializeWeights
	
	private double randomBias() {
		return Math.random()-0.5 * 2.0;
	}// end method randomBias
}// end class Network
