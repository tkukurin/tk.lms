"""showcase: penzai demo, inspecting Jax model internals.

This is mainly c/p from their [tutorials] with a few additions.

[tutorials]: https://penzai.readthedocs.io/en/stable/notebooks/how_to_think_in_penzai.html
"""

# %%
import penzai
from penzai import pz
# %%
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
# %%
import jax
from penzai.models import simple_mlp

mlp = simple_mlp.MLP.from_config(
    name="mlp",
    init_base_rng=jax.random.key(0),
    feature_sizes=[8, 32, 32, 8]
)

mlp
# %%
# if you copy this output, you get the same model
unbound_mlp, _ = pz.unbind_params(mlp)
unbound_mlp
# %%
mlp(pz.nx.ones({"features": 8}))
# %% access model internals
mlp.sublayers[2].sublayers[0]
# %%

copied = penzai.models.simple_mlp.MLP( # Sequential
  sublayers=[
    penzai.nn.linear_and_affine.Affine( # Sequential
      sublayers=[
        penzai.nn.linear_and_affine.Linear(weights=penzai.core.variables.ParameterSlot(label='mlp/Affine_0/Linear.weights'), in_axis_names=('features',), out_axis_names=('features_out',)),
        penzai.nn.linear_and_affine.RenameAxes(old=('features_out',), new=('features',)),
        penzai.nn.linear_and_affine.AddBias(bias=penzai.core.variables.ParameterSlot(label='mlp/Affine_0/AddBias.bias'), new_axis_names=()),
      ],
    ),
    penzai.nn.basic_ops.Elementwise(fn=jax.nn.relu),
    penzai.nn.linear_and_affine.Affine( # Sequential
      sublayers=[penzai.nn.linear_and_affine.Linear(weights=penzai.core.variables.ParameterSlot(label='mlp/Affine_1/Linear.weights'), in_axis_names=('features',), out_axis_names=('features_out',)), penzai.nn.linear_and_affine.RenameAxes(old=('features_out',), new=('features',)), penzai.nn.linear_and_affine.AddBias(bias=penzai.core.variables.ParameterSlot(label='mlp/Affine_1/AddBias.bias'), new_axis_names=())],
    ),
    penzai.nn.basic_ops.Elementwise(fn=jax.nn.relu),
    penzai.nn.linear_and_affine.Affine( # Sequential
      sublayers=[penzai.nn.linear_and_affine.Linear(weights=penzai.core.variables.ParameterSlot(label='mlp/Affine_2/Linear.weights'), in_axis_names=('features',), out_axis_names=('features_out',)), penzai.nn.linear_and_affine.RenameAxes(old=('features_out',), new=('features',)), penzai.nn.linear_and_affine.AddBias(bias=penzai.core.variables.ParameterSlot(label='mlp/Affine_2/AddBias.bias'), new_axis_names=())],
    ),
  ],
)

copied == unbound_mlp
# %% insert intermediate variables, e.g. can be useful for printing state
from typing import Any

@pz.pytree_dataclass
class AppendIntermediate(pz.nn.Layer):
  saved: pz.StateVariable[list[Any]]
  def __call__(self, x: Any, **unused_side_inputs) -> Any:
    self.saved.value = self.saved.value + [x]
    return x

var = pz.StateVariable(value=[], label="my_intermediates")

saving_model = (
    pz.select(mlp)
    .at_instances_of(pz.nn.Elementwise)
    .insert_after(AppendIntermediate(var))
)

output = saving_model(pz.nx.ones({"features": 8}))
intermediates = var.value
intermediates

# %% there is some way to also build a penzai-powered transformer.
import jax.numpy as jnp
from penzai.models import transformer

transformer.variants.gpt_neox.build_gpt_neox_transformer(
    transformer.variants.gpt_neox.GPTNeoXTransformerConfig(
        num_attention_heads=4,
        embedding_dim=128,
        projection_dim=32,
        mlp_hidden_dim=128,
        num_decoder_blocks=4,
        activation_fn='gelu_approx',
        rope_wavelength=1.0,
        parameter_dtype=jnp.int32,
        activation_dtype=jnp.float16,
        vocab_size=32,
        rope_subset_size=32,
        layernorm_epsilon=1e-5,
    )
)