import java.io.*;

/**
 * @author Pree Thiengburanthum
 * CSC5542 Neural Networks
 * BP.java
 */
public class BP {
	public static int maxEpochs = 10000;
	public static int maxTrain = 1000;
	public static int maxTest = 100;
	
	public static double[] stepSize = {0.1, 0.2, 0.3, 0.4, 0.5};
	public static double threshold = 0.1;
	public static double weightRange = 0.001;
	public static double momentum ;
	
	public static void main(String args[]) throws IOException {
	
		for(int i=0; i<stepSize.length; i++) {
			System.out.println("MaxEpochs: " + maxEpochs + " maxTrain: " + maxTrain + " maxTest: " 
					+ maxTest + " stepSize: " + stepSize[i] + " threshold: " + threshold + 
					" weightRange" + weightRange + " momentum: " + momentum);
			Network myBP = new Network(maxEpochs, maxTrain, maxTest, stepSize[i], threshold, weightRange, momentum);
		
			myBP.initializeNetwork();
			myBP.runSim();
			
			BufferedWriter out;
			out = new BufferedWriter(new FileWriter(stepSize[i] + "TrainPoints.csv"));
			for(int j=0; j<myBP.getTrainingPoints().size(); j++)
				out.write(myBP.getTrainingPoints().elementAt(j) + "\n");
			out.close();
			
			out = new BufferedWriter(new FileWriter(stepSize[i] + "TestResults.csv"));
			for(int j=0; j<myBP.getTestPoints().size(); j++)
				out.write(myBP.getTestPoints().elementAt(j) + "\n");
			out.close();
			
			out = new BufferedWriter(new FileWriter(stepSize[i] + "SuccessRate.csv"));
			out.write(myBP.getSuccessRates());
			out.close();
			
			
			
			out = new BufferedWriter(new FileWriter(stepSize[i] + "SSError.csv"));
			for(int j=0; j<myBP.getSSError().size(); j++)
				out.write(myBP.getSSError().elementAt(j) + "\n");
			out.close();
			
			
			System.out.println("converged at epoch " + myBP.getConverged());
			System.out.println("Success rate " + myBP.getFinalSuccessRate() + "%");
			System.out.println();
		}
		
	}// end main
}// end class BP
