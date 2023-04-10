import Value from "./engine.mjs";

class Neuron {
  constructor(inputSize) {
    this.weights = Array.from({ length: inputSize }, () => new Value(Math.random() * 2 - 1));
    this.bias = new Value(Math.random() * 2 - 1);
  }
  calc(input) {
    let dotp = this.bias;
    for (let i = 0; i < input.length; i++) {
      dotp.data += this.weights[i].data * input[i];
    }
    return dotp.tanh();
  }
  params() {
    return this.weights.concat(this.bias);
  }
}

class Layer {
  constructor(inputSize, outputSize) {
    this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
  }
  calc(input) {
    const out = this.neurons.map((neuron) => neuron.calc(input));
    return out.length === 1 ? out[0] : out;
  }
  params() {
    return this.neurons.flatMap((neuron) => neuron.params());
  }
}

class MLP {
  constructor(inputSize, outputSize) {
    this.layers = Array.from(
      { length: outputSize.length },
      (layer, index) => new Layer(inputSize, outputSize[index])
    );
  }
  calc(input) {
    return this.layers.map((layer) => layer.calc(input)).pop();
  }
  params() {
    return this.layers.flatMap((layer) => layer.params());
  }
}
