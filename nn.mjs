import Value from "./engine.mjs";

export class Neuron {
  constructor(inputSize) {
    this.weights = Array.from({ length: inputSize }, () => new Value(Math.random() * 2 - 1));
    this.bias = new Value(Math.random() * 2 - 1);
  }
  calc(input) {
    let dotp = this.bias;
    for (let i = 0; i < this.weights.length; i++) {
      dotp = dotp.add(this.weights[i].mul(input[i]));
    }
    return dotp.tanh();
  }
  params() {
    return this.weights.concat(this.bias);
  }
}

export class Layer {
  constructor(inputSize, outputSize) {
    this.neurons = Array.from({ length: outputSize }, () => new Neuron(inputSize));
  }
  calc(input) {
    const out = this.neurons.map((neuron) => neuron.calc(input));
    return out.length === 1 ? out[out.length - 1] : out;
  }
  params() {
    return this.neurons.flatMap((neuron) => neuron.params());
  }
}

export class MLP {
  constructor(inputSize, outputSize) {
    const sz = [inputSize, ...outputSize];
    this.layers = outputSize.map((output, i) => new Layer(sz[i], sz[i + 1]));
  }
  calc(input) {
    for (const layer of this.layers) {
      input = layer.calc(input);
    }
    return input;
  }
  params() {
    return this.layers.flatMap((layer) => layer.params());
  }
}
