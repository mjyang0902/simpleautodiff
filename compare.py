from simpleautodiff.simpleautodiff import mul, sin, add, forward, reverse_mode, Node, sub
import time
import math
import numpy as onp         
import autograd.numpy as np 
from autograd import grad

def calculate_final_mse(w, b, X, Y):
    z = onp.dot(X, w) + b
    preds = 1.0 / (1.0 + onp.exp(-z))
    mse = onp.mean((preds - Y) ** 2)
    return mse

N_SAMPLES = 1000
N_FEATURES = 10
EPOCHS = 100
lr = 0.05

print(f"--- Data Generation (Samples: {N_SAMPLES}, Features: {N_FEATURES}) ---")
onp.random.seed(42)
X_data = onp.random.randn(N_SAMPLES, N_FEATURES)
true_weights = onp.random.randn(N_FEATURES)
logits = onp.dot(X_data, true_weights)
Y_data = (logits > 0).astype(float)

def run_hips():
    print("Running HIPS Autograd...", end="", flush=True)
    w = np.zeros(N_FEATURES)
    b = 0.0
    
    def loss_func(w, b, x_vec, y_scalar):
        z = np.dot(x_vec, w) + b
        y_pred = 1.0 / (1.0 + np.exp(-z))
        diff = y_pred - y_scalar
        return diff * diff

    grad_loss = grad(loss_func, argnum=(0, 1))

    start = time.perf_counter()
    for j in range(EPOCHS):
        for i in range(N_SAMPLES):
            dw, db = grad_loss(w, b, X_data[i], Y_data[i])
            w -= lr * dw
            b -= lr * db

    end = time.perf_counter()
    final_loss = calculate_final_mse(w, b, X_data, Y_data)
    print(" Done.")
    return end - start, final_loss, w, b

def run_simple():
    print("Running Simpleautodiff Autograd...", end="", flush=True)
    w_vals = [0.0] * N_FEATURES
    b_val = 0.0
    
    start = time.perf_counter()
    for j in range(EPOCHS):
        for i in range(N_SAMPLES):
            x_nodes = [Node(val) for val in X_data[i]]
            y_node = Node(Y_data[i])
            
            w_nodes = [Node(val) for val in w_vals]
            b_node = Node(b_val)
            
            acc = mul(w_nodes[0], x_nodes[0])
            for k in range(1, N_FEATURES):
                term = mul(w_nodes[k], x_nodes[k])
                acc = add(acc, term)
                
            z = add(acc, b_node)
            
            sig_val = 1.0 / (1.0 + math.exp(-z.value)) if -z.value < 700 else 0.0
            s_node = Node(sig_val, [z], "sigmoid")
            s_node.grad_wrt_parents = [sig_val * (1.0 - sig_val)]
            z.child_nodes.append(s_node)
            
            y_pred = s_node
            
            diff = sub(y_pred, y_node)
            loss = mul(diff, diff)
            
            reverse_mode(loss)
            
            for k in range(N_FEATURES):
                w_vals[k] -= lr * w_nodes[k].reverse_partial_derivative
            b_val -= lr * b_node.reverse_partial_derivative

    end = time.perf_counter()
    
    final_w_np = onp.array(w_vals)
    final_loss = calculate_final_mse(final_w_np, b_val, X_data, Y_data)
    
    print(" Done.")
    return end - start, final_loss, final_w_np, b_val

hips_time, hips_loss, hips_w, hips_b = run_hips()
simple_time, simple_loss, simple_w, simple_b = run_simple()

print("\n" + "="*50)
print(f"{'Metric':<15} | {'HIPS':<15} | {'Simpleautodiff':<15}")
print("-" * 50)
print(f"{'Time (sec)':<15} | {hips_time:<15.4f} | {simple_time:<15.4f}")
print(f"{'Final Loss':<15} | {hips_loss:<15.6f} | {simple_loss:<15.6f}")
print("-" * 50)

print(f"HIPS are faster than simpleautodiff by a factor of {simple_time / hips_time:.2f}x")
loss_diff = abs(hips_loss - simple_loss)
print(f"\nLoss Difference: {loss_diff:.10f}")

if loss_diff < 1e-5:
    print("The Loss values are very close. Gradient implementation is likely correct. ✅")
else:
    print("The Loss values differ significantly. There may be an issue with the gradient implementation. ❌")


print("\n Sample Weights Comparison (first 3 weights):")
print(f"HIPS:   {hips_w[:3]}")
print(f"Simpleautodiff: {simple_w[:3]}")
