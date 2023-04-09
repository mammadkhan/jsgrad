class Value {
  constructor(data, children, op) {
    this.data = data;
    this.grad = 0.0;
    this.op = op;
    this.backward = function () {};
    this.children = children;
  }
  //Basic operations
  add(other) {
    let out = new Value(this.data + other.data, [this, other], "+");

    out.backward = () => {
      this.grad += 1.0 * out.grad;
      other.grad += 1.0 * out.grad;
    };

    return out;
  }
  multiply(other) {
    let out = new Value(this.data * other.data, [this, other], "*");

    out.backward = () => {
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };

    return out;
  }
  //Activation
  tanh() {
    let out = new Value(Math.tanh(this.data), [this], "tanh");

    out.backward = () => {
      this.grad += (1 - out.data ** 2) * out.grad;
    };

    return out;
  }
  //Backpropagation
  backProp() {
    //Sorting graph in topological order
    let sorted = [];
    let visited = new Set();
    function topological_sort(node) {
      if (visited.has(node)) return;
      visited.add(node);
      if (!node.children) sorted.push(node);
      else {
        for (let child of node.children) {
          topological_sort(child);
        }
        sorted.push(node);
      }
    }

    topological_sort(this);
    sorted.reverse();

    this.grad = 1.0;
    //calling backward on each node
    for (let node of sorted) {
      node.backward();
    }
  }
}

class Neuron {
  constructor(inputSize) {
    this.weights = [];
    this.bias = new Value(Math.random() * (1 - -1) + -1);
    for (let i = 0; i < inputSize; i++) {
      this.weights.push(new Value(Math.random() * (1 - -1) + -1));
    }
  }
  init(input) {
    let dotProduct = new Value(this.bias.data);
    for (let i = 0; i < input.length; i++) {
      dotProduct.data += input[i] * this.weights[i].data;
    }
    return dotProduct.tanh();
  }
}

class Layer {
  constructor(inputSize, outputSize) {
    this.neurons = [];
    for (let i = 0; i < outputSize; i++) {
      this.neurons.push(new Neuron(inputSize));
    }
  }
  init(input) {
    let outputs = [];
    for (let i = 0; i < this.neurons.length; i++) {
      outputs.push(this.neurons[i].init(input));
    }
    return outputs;
  }
}

class MLP {
  constructor(inputSize, outputSize) {
    this.layers = [];
    for (let i = 0; i < outputSize.length; i++) {
      this.layers.push(new Layer(inputSize, outputSize[i]));
    }
  }

  init(input) {
    let outputs = [];
    for (let i = 0; i < this.layers.length; i++) {
      outputs.push(this.layers[i].init(input));
    }
    return outputs;
  }
}

const rand = [2, 1, 6, 4];
const n = new MLP(4, [4, 4, 1]);
console.log(n.init(rand));
