# This README is outdated, this repo has advanced a lot lol I'll update this later
# tsgrad

A small PyTorch-like ML framework built from scratch in TypeScript, with no external ML libraries, all math implemented by hand.

Inspired by [@srush_nlp](https://x.com/srush_nlp)'s [minitorch](https://github.com/minitorch/minitorch).  
  
Built from scratch to learn how ml frameworks/models work.  

---

## What's implemented

- **Scalar** - scalar values that track their own computation history
- **Reverse-mode autograd** - automatic differentiation via a dynamically built computation graph
- **Backpropagation** - gradient computation via reverse traversal of the graph
- **Parameters & Modules** - base classes for building trainable components
- **Neural network layers** - `Linear` layers with configurable activation functions (`relu`, `sigmoid`)
- **SGD / SGD + momentum** - gradient descent optimizers

## Example

```typescript
export class SimpleNetwork extends Network {
  constructor() {
    super();
    this.addLayer(Linear, "relu", 2, 3);
    this.addLayer(Linear, "sigmoid", 3, 1);
  }
}
```
  
# To download and test on the mnist dataset run the following from inside framework
```bash
mkdir project\data\mnist

curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -o project\data\mnist\train-images-idx3-ubyte.gz
curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -o project\data\mnist\train-labels-idx1-ubyte.gz
curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz -o project\data\mnist\t10k-images-idx3-ubyte.gz
curl.exe -L https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz -o project\data\mnist\t10k-labels-idx1-ubyte.gz
```  
  
## Roadmap  
Working on tensors now, then CUDA next.  
