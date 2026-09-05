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

## Roadmap  
Working on tensors now, then CUDA next.  
