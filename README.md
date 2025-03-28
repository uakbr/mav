

![image](https://github.com/user-attachments/assets/7147a0d7-d8ad-4b40-bb53-270c4b7afceb)


model activation visualiser 

### installation

if uv is installed:

`uv run --with git+https://github.com/attentionmech/mav@pilot mav --model gpt2 --prompt "hello mello"`

without uv:

0. venv setup and activate
1. `pip install git+https://github.com/attentionmech/mav@pilot`
2. `mav --model gpt2 --prompt "hello mello"`

also:

`dev` branch's first tag is mapped to `openmav`. so you can also do `pip install openmav`

(can replace with other hf models like `meta-llama/Llama-3.2-1B` or `HuggingFaceTB/SmolLM-135M`)


### what are we plotting

at every point in prediction, there are many next tokens which are possible; with different amount of confidence. that is what tokens and the numbers near them represent.

activations are basically numbers which represent the forward pass over the network during inference. layer-wise activations means we are looking at them per layer basis. generally, a layer/block in gpt style models contain MLP sub-block and a attention sub-block. we are plotting the values as they come out of MLP sub-block for a layer while taking their l2-norm (please study l2 norm). there are other options also like average or max etc. which don't really capture any information which is easy to reason about.

For the attention sub-block, what we measure is entropy. attention in transformer architecture is about how tokens affect other tokens. the entropy values loosely can mean something like how wide the attending is. i.e. if it's a sharp token to token relation or it's a net cast over a wide area. again, analogies can be misleading. do study this stuff from multiple sources/videos to not form wrong mental models.
