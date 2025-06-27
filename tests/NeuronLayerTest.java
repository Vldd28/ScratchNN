public class NeuronLayerTest {
    public static void main(String[] args){
        Matrix weights = new Matrix(new double[][] {
                {0.2, 0.8, -0.5},
                {0.5, -0.91, 0.26}
        });

        Matrix biases = new Matrix(new double[][] {
                {2.0},
                {3.0}
        });

        NeuronLayer layer = new NeuronLayer(weights, biases);
        Matrix input = new Matrix(new double[][] {
                {-1.0},
                {-2.0},
                {-3.0}
        });

        Matrix output = layer.forward(input);

        System.out.println("Output from NeuronLayer forward():");
        for(int i = 0; i < output.rows; i++)
        {
            for(int j = 0; j < output.cols; j++)
                System.out.println(output.data[i][j]);
        }
    }

    //empty matrix, error

    //empty bias and weight, works

    //negative values works

}
