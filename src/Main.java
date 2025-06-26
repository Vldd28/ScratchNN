public class Main {
	public static void main(String[] args) {
		double[][] weightsArray = {
				{0.2, 0.4, 0.6},
				{0.5, 0.1, 0.3}
		};
		Matrix weights = new Matrix(weightsArray);
		double[][] biasesArray = {
				{0.1},
				{0.2}
		};
		Matrix biases = new Matrix(biasesArray);
		NeuronLayer layer = new NeuronLayer(weights, biases);
		double[][] inputArray = {
				{1.0},
				{2.0},
				{3.0}
		};
		Matrix input = new Matrix(inputArray);
		Matrix output = layer.forward(input);
		System.out.println("Output:");
		for (int i = 0; i < output.rows; i++) {
			System.out.println(output.data[i][0]);
		}

		System.out.println("testing the transpose");
		Matrix trans = new Matrix(biasesArray);
		trans.transpose();
		for(int i = 0; i < trans.rows; i++)
		{
			for(int j = 0; j < trans.cols; j++)
				System.out.println(trans.data[i][j]);
		}
	}
}
