public class ReLU implements ActivationFunction{
    public Matrix apply(Matrix input) {
        Matrix output = input;
        for(int i =0;i<input.rows;i++)
            for (int j = 0 ;j < input.cols;j++)
                output.data[i][j] = Math.max(0,input.data[i][j]);
        return output;
    }

    public Matrix derivative(Matrix input) {
        Matrix output = input;
        for (int i =0;i<input.rows;i++)
            for(int j = 0 ;j < input.cols;j++)
                if(output.data[i][j] > 0){
                    output.data[i][j] = 1;
                }
            else
            output.data[i][j] = 0;
        return output;
    }
}
