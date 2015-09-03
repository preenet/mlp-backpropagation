/**
 * @author Pree Thiengburanthum
 * CSC5542 Neural Networks
 * Point.java
 */

public class Point {
	private double x;
	private double y;
	private char type;
	private boolean success;
	
	public Point() {
		this.x = (Math.random() * 4) - 2;
		this.y = (Math.random() * 4) - 2;
		
		// identify the type
		if((Math.pow(this.x, 2) + Math.pow(this.y, 2)) > 1 ) 
			this.type = 'B';
		else
			this.type = 'A';
		
		this.success = false;
	}// end constructor
	
	public Point(double x, double y) {
		this.x = x;
		this.y = y;
		// identify the type
		if((Math.pow(this.x, 2) + Math.pow(this.y, 2)) > 1 ) 
			this.type = 'B';
		else
			this.type = 'A';
	}
	
	public double getX() {
		return this.x;
	}// end method getX
	
	public double getY() {
		return this.y;
	}// end method getY
	
	public char getType() {
		return this.type;
	}// end method getType
	
	public void setType(char t) {
		this.type = t;
	}
	
	public void setSuccess() {
		this.success = true;
	}// end method setSuccess

	public String toString() {
		if(!this.success)
			return this.type + ", " + this.x + ", " + this.y;
		return this.type + ", " + this.x + ", " + this.y + ", success";
	}// end override method toString
}// end class Point
