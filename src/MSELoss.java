public class MSELoss implements LossFunction {
	public double compute(Matrix output, Matrix target){
		double sum = 0.0;
		for (int i = 0 ;i < output.rows; i++){
			for (int j = 0; j < output.cols ; j++){
				double diff = output.data[i][j] - target.data[i][j];
				sum += diff*diff;
			}
		}
		return sum / (output.rows * output.cols);
	}
}
