  High Impact Options

  1. GPU Support

  The FFI has ts_cuda_is_available() but full CUDA tensor operations may need work.
  - Would dramatically improve training speed
  - Essential for larger models
  - Requires testing tensor device transfers (.cuda(), .to())

  2. Model Serialization

  Save and load trained models:
  - torch.save(model, 'model.pt')
  - torch.load('model.pt')
  - Essential for any real usage

  3. More NN Modules

  - RNN/LSTM/GRU - Sequence modeling
  - Embedding - NLP tasks
  - MultiheadAttention/Transformer - Modern architectures
  - Flatten, Unflatten - Shape manipulation layers

  4. Advanced Tensor Operations

  - einsum - Flexible tensor contractions
  - Advanced indexing (tensor[mask], tensor[:, 1:3])
  - scatter, gather - Index-based operations
  - cat, stack, split - Tensor manipulation

  5. Documentation & Examples

  - API reference docs
  - More training examples (text classification, image segmentation)
  - Tutorial notebooks

  Medium Impact Options

  - Learning rate schedulers (may already exist partially)
  - Gradient clipping
  - Mixed precision training (fp16)
  - Data augmentation utilities
  - Model summary (print layer shapes/params)

  What's your priority?

  What are you most interested in building with the library? That would help narrow down what to tackle next:

  - Computer Vision → More conv layers, augmentation, pretrained models
  - NLP → Embedding, RNN/LSTM, Attention
  - RL → Focus on inference speed, small model optimization
  - General research → Model save/load, more tensor ops
  - Production deployment → ONNX export, quantization