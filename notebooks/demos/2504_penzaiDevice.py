"""Penzai demo: move to GPU.
"""
# %%
import jax
from penzai import pz

def get_device_str(x):
    if hasattr(x, 'device') and x.device is not None:
        return str(x.device)
    elif hasattr(x, 'devices') and x.devices:
        return str(x.devices)
    return "unknown"

def check_device(model):
    """Check device placement of model parameters"""
    slot_tree, params = pz.unbind_params(model)
    param_arrays = {param.label: param.value for param in params}
    for name, array in param_arrays.items():
        if hasattr(array, 'data_array'):  # For NamedArrays
            device = get_device_str(array.data_array)
        else:
            device = get_device_str(array)
        print(f"{name} is on {device}")
    return slot_tree, params

model = pz.nn.Linear.from_config(
    name="linear", 
    init_base_rng=jax.random.key(0),
    input_axes={"embedding": 10},
    output_axes={"output": 5}
)

print("Initial placement:")
slot_tree, params = check_device(model)

if len(jax.devices("gpu")) > 0:
    gpu_params = jax.device_put([param.freeze() for param in params], jax.devices("gpu")[0])
    gpu_model = pz.bind_variables(slot_tree, [p.unfreeze_as_copy() for p in gpu_params])
    
    print("\nAfter moving to GPU:")
    check_device(gpu_model)
    
    cpu_params = jax.device_put(gpu_params, jax.devices("cpu")[0])
    cpu_model = pz.bind_variables(slot_tree, [p.unfreeze_as_copy() for p in cpu_params])
    
    print("\nAfter moving back to CPU:")
    check_device(cpu_model)
else:
    print("No GPU available")
# %%
import jax
from penzai import pz

def model_to_device(model: pz.nn.Layer, device: jax.Device) -> pz.nn.Layer:
    """Move a Penzai model to a specific device"""
    stateless_model, variables = pz.unbind_variables(model, freeze=True)
    variables_on_device = jax.device_put(variables, device)
    return pz.bind_variables(stateless_model, variables_on_device, unfreeze_as_copy=True)

model = pz.nn.Linear.from_config(
    name="linear",
    init_base_rng=jax.random.key(0),
    input_axes={"embedding": 10},
    output_axes={"output": 5}
)

cpu_device = jax.devices("cpu")[0]
model_on_cpu = model_to_device(model, cpu_device)
print(f"Model moved to CPU: {cpu_device}")

if len(jax.devices("gpu")) > 0:
    gpu_device = jax.devices("gpu")[0]
    model_on_gpu = model_to_device(model, gpu_device)
    print(f"Model moved to GPU: {gpu_device}")
    
    # Create some input data on GPU
    x = pz.nx.ones({"batch": 2, "embedding": 10})
    x_gpu = jax.device_put(x, gpu_device)
    
    from penzai.core.named_axes import NamedArray
    result_gpu: NamedArray = model_on_gpu(x_gpu)
    print(f"Computed result on GPU with shape: {result_gpu.named_shape}")
    
    result_cpu = jax.device_put(result_gpu, cpu_device)
    print(f"Result moved back to CPU")
else:
    print("No GPU available")

# %%
