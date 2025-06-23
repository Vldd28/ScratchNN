import java.util.List;

public class NeuralNetwork {
	List<NeuronLayer> layers;
	double learningRate;
	public NeuralNetwork(List<NeuronLayer> layers){
		this.layers = layers;
	}
	public NeuralNetwork(List<NeuronLayer> layers, double learningRate){
		this.layers = layers;
		this.learningRate = learningRate;
	}
	public Matrix predict(Matrix input){
		Matrix output = input;
		for (NeuronLayer layer:layers){
			output = layer.forward(output);
		}
		return output;
	}
}
