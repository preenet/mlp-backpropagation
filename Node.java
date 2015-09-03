/**
 * @author Pree Thiengburanthum
 * CSC5542 Neural Networks
 * Node.java
 */

public class Node {
	private double activation;
	private double bias;
	private double oldBias;
	
	public Node(double a, double b) {
		this.activation = a;
		this.bias = b;
	}// end constructor
	
	public void setActivation(double a) {
		this.activation = a;
	}// end method setActivation
	
	public double getActivation() {
		return this.activation;
	}// end method getActivation
	
	public void setBias(double b) {
		this.bias = b;
	}// end method setBias
	
	public double getBias() {
		return this.bias;
	}// end method getBias
	
	public void setOldBias(double b) {
		this.oldBias = b;
	}// end method setOldBias
	
	public double getOldBias() {
		return this.oldBias;
	}// end method getOldBias
	
	public String toString() {
		return this.activation + " " + this.bias;
	}// end override method toString
}// end class Node
