public class MSELossTest {
    public static void main(String[] args) {
        MSELoss loss = new MSELoss();

        Matrix output = new Matrix(new double[][]{
                {1.0, 2.0},
                {7.0, 9.0},
        });

        Matrix target = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
        });

        double result = loss.compute(output, target);
        System.out.println(result);
    }
}