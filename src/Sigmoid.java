public class Sigmoid implements ActivationFunction {
    public Matrix apply(Matrix input) {
        Matrix output = new Matrix(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = 1.0 / (1.0 + Math.exp(-input.data[i][j]));
            }
        }
        return output;
    }

    public Matrix derivative(Matrix input) {
        Matrix sig = this.apply(input);
        Matrix oneMinusSig = new Matrix(sig.rows, sig.cols);
        for (int i = 0; i < sig.rows; i++) {
            for (int j = 0; j < sig.cols; j++) {
                oneMinusSig.data[i][j] = 1.0 - sig.data[i][j];
            }
        }
        return sig.elementWiseMultiply(oneMinusSig);
    }
}
