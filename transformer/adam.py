import numpy as np


class Adam:
    def __init__(self, model, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=1.0):
        self.model = model
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_value = clip_value

        self.t = 0
        self.m = {}  # First moment estimates (momentum)
        self.v = {}  # Second moment estimates (RMSprop)

    def step(self):
        self.t += 1
        self._step_recursive(self.model)

    def _step_recursive(self, module):
        # Iterate over all attributes of the current module
        for attr_name in vars(module):
            attr = getattr(module, attr_name)

            # 1. If the attribute is a sub-module we defined, recurse into it
            if type(attr).__name__ in ['SelfAttention', 'FeedForward', 'Encoder', 'LMHead', 'Linear', 'Embedding',
                                       'LayerNorm', 'Transformer']:
                self._step_recursive(attr)

            # 2. If the attribute is a NumPy array, it might be a weight we need to update
            elif isinstance(attr, np.ndarray):
                # Determine the gradient name (handles both dWQ and d_weights conventions)
                grad_name = None
                if hasattr(module, f'd_{attr_name}'):
                    grad_name = f'd_{attr_name}'
                elif hasattr(module, f'd{attr_name}'):
                    grad_name = f'd{attr_name}'

                if grad_name:
                    grad = getattr(module, grad_name)

                    # Apply Gradient Clipping to prevent explosions
                    if self.clip_value > 0:
                        np.clip(grad, -self.clip_value, self.clip_value, out=grad)

                    # Create a unique key for this specific parameter matrix
                    key = (id(module), attr_name)

                    # Initialize moments if they don't exist yet
                    if key not in self.m:
                        self.m[key] = np.zeros_like(attr)
                        self.v[key] = np.zeros_like(attr)

                    # Update biased first moment estimate
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                    # Update biased second raw moment estimate
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

                    # Compute bias-corrected first moment estimate
                    m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                    # Compute bias-corrected second raw moment estimate
                    v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                    # Update the weight IN PLACE (this directly modifies module.WQ, module.weights, etc.)
                    attr -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        self._zero_grad_recursive(self.model)

    def _zero_grad_recursive(self, module):
        for attr_name in vars(module):
            attr = getattr(module, attr_name)

            # Recurse into sub-modules
            if type(attr).__name__ in ['SelfAttention', 'FeedForward', 'Encoder', 'LMHead', 'Linear', 'Embedding',
                                       'LayerNorm', 'Transformer']:
                self._zero_grad_recursive(attr)

            # Zero out gradient arrays based on your naming conventions
            elif isinstance(attr, np.ndarray) and (
                    attr_name.startswith('d_') or
                    attr_name.startswith('dW') or
                    attr_name.startswith('dweights') or
                    attr_name.startswith('dbiases') or
                    attr_name.startswith('d_gamma') or
                    attr_name.startswith('d_beta')
            ):
                attr.fill(0)