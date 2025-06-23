import java.lang.Math;

public class NeuronLayer {
	Matrix weights;
	Matrix biases;
	public NeuronLayer(Matrix W, Matrix b){
		weights = W;
		biases = b;
	}
	public Matrix forward(Matrix input){
		Matrix result = weights.multiply(input);
		result.addVectors(biases);
		return result;
	}
	private Matrix activate(Matrix m) {
		Matrix activated = new Matrix(m.rows, m.cols);
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				activated.data[i][j] = relu(m.data[i][j]);
			}
		}
		return activated;
	}
	
	private double relu(double x) {
		return Math.max(0, x);
	}
	private double sigmoid(double x){
		return (1/(1 + Math.exp(-x)));
	}
	
}
