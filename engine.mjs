export default class Value {
  constructor(data, children = null, op = null) {
    this.data = data;
    this.grad = 0.0;
    this.op = op;
    this.backward = () => {};
    this.children = children;
  }
  //Operations
  add(other) {
    const assertOther = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + assertOther.data, [this, assertOther], "+");

    out.backward = () => {
      this.grad += 1.0 * out.grad;
      assertOther.grad += 1.0 * out.grad;
    };

    return out;
  }
  mul(other) {
    const assertOther = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * assertOther.data, [this, assertOther], "*");

    out.backward = () => {
      this.grad += other.data * out.grad;
      assertOther.grad += this.data * out.grad;
    };

    return out;
  }
  pow(other) {
    const assertOther = typeof other == "number" && other;
    const out = new Value(Math.pow(this.data, assertOther), [this], `**${assertOther}`);

    out.backward = () => {
      this.grad = assertOther * Math.pow(this.data, assertOther - 1) * out.grad;
    };

    return out;
  }
  //Activation
  tanh() {
    const out = new Value(Math.tanh(this.data), [this], "tanh");

    out.backward = () => {
      this.grad = (1 - out.data ** 2) * out.grad;
    };

    return out;
  }
  //Backpropagation
  backprop() {
    let sorted = [];
    let visited = new Set();
    function topological_sort(node) {
      if (visited.has(node)) return;
      visited.add(node);
      if (!node.children) {
        sorted.push(node);
        return;
      }
      for (let child of node.children) {
        topological_sort(child);
      }
      sorted.push(node);
    }

    topological_sort(this);
    sorted.reverse();

    this.grad = 1.0;

    for (let node of sorted) {
      node.backward();
    }
  }
}
