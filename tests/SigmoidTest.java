public class SigmoidTest {
    public static void main(String[] args)
    {
        Sigmoid sigmoid = new Sigmoid();

        Matrix input = new Matrix(new double[][]{
                {1.0, 1.0},
                {1.0, 1.0}
        });
        Matrix applied = sigmoid.apply(input);
        System.out.println("applied sigmoid:");
        for(int i = 0; i < applied.rows; i++)
        {
            for(int j = 0; j < applied.cols; j++)
                System.out.println(applied.data[i][j]);
        }

        Matrix derivative = sigmoid.derivative(input);
        System.out.println("applied derivatie: ");
        for(int i = 0; i < derivative.rows; i++)
        {
            for(int j = 0; j < derivative.cols; j++)
                System.out.println(derivative.data[i][j]);
        }
    }

    //{-2.0, 0.0, 2.0},
    //{5.0, -5.0, 1.0} passed

    //{0.0} passed

    //{1.0, 5.0, 10.0} passed

    //{-1.0, -5.0, -10.0} passed

    //{-1000.0, 0.0, 1000.0} passed

    //empty error same as relu, should return 0x0

    //{1.0, 2.0, 3.0},
    //{4.0, 5.0, 6.0} passed

    //{0.1, 0.5, 0.9} passed

    //{1.0, 1.0},
    //{1.0, 1.0} passed


}
