public interface ActivationFunction {
    Matrix apply(Matrix input);
    Matrix derivative(Matrix input);
}
