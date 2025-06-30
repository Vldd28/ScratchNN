public class ReLUTest {
    public static void main(String[] args)
    {
        ReLU relu = new ReLU();

        Matrix applied = relu.apply(new Matrix(new double[][]{

        }));

        System.out.println("relu apply says: ");
        for(int i = 0; i < applied.rows; i++)
        {
            for(int j = 0; j < applied.cols; j++)
            {
                System.out.println(applied.data[i][j]);
            }
        }

        Matrix derivative = relu.derivative(new Matrix(new double[][]{

        }));
        System.out.println("relu derivative says: ");
        for(int i = 0; i < derivative.rows; i++)
        {
            for(int j = 0; j < derivative.cols; j++)
            {
                System.out.println(derivative.data[i][j]);
            }
        }

    }

    //{0.0, 0.0},
    //{0.0, 0.0}  passed

    //{-1.0, -0.1},
    //{-100.0, -999.9} passed

    //{1.0, 2.5},
    //{100.0, 0.0001} passed

    //{-3.0, 0.0, 4.0} passed

    //empty matrix failed, should return a 0x0 matrix

    //{-1e10, 1e10},
    //{-1e-10, 1e-10} passed

    //{1.0, -1.0, 0.0},
    //{-0.5, 2.0, -3.0} passed

}
