public class Matrix {
	double[][] data;
	int rows;
	int cols;
	
	public Matrix(int rows, int cols) {
		data = new double[rows][cols];
		this.rows = rows;
		this.cols = cols;
	}
	
	public Matrix(double[][] inputData) {
		this.rows = inputData.length;
		this.cols = inputData[0].length;
		
		data = new double[rows][cols];
		for (int i = 0; i < rows; i++) {
			System.arraycopy(inputData[i], 0, data[i], 0, cols);
		}
	}
	public double[] getCol(int col){
		double[] result = new double[rows];
		for (int i = 0 ;i < rows ;i++){
			result[i] = data[i][col];
		}
		return result;
	}
	public Matrix multiply(Matrix other) {
		if (this.cols != other.rows) {
			throw new IllegalArgumentException("Incompatible matrix sizes for multiplication.");
		}
		Matrix result = new Matrix(this.rows, other.cols);
		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < other.cols; j++) {
				result.data[i][j] = dotProduct(this.data[i], other.getCol(j));
			}
		}
		return result;
	}
	public static double dotProduct(double[]v1, double[]v2){
		if (v1.length != v2.length){
			throw new IllegalArgumentException("Vectors must have the same length.");
		}
		double sum = 0.0;
		for (int i = 0;i < v1.length; i++){
			sum += v1[i] * v2[i];
		}
		return sum;
	}
	public void addVectors(Matrix other){
		if (this.rows != other.rows || other.cols != 1 || this.cols != 1) {
			throw new IllegalArgumentException("Matrix dimensions must match for vector addition.");
		}
		for (int i = 0; i < other.rows; i++){
			data[i][0] += other.data[i][0];
		}
	}
	
}
