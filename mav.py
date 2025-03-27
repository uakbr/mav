# /// script
# dependencies = [
#   "torch",
#   "rich",
#   "torch",
#   "numpy",
#   "transformers"
# ]
# ///

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
from rich.console import Console
from rich.text import Text
from rich.layout import Layout
from rich.panel import Panel
import argparse


class ModelActivationVisualizer:
    def __init__(self, model_name, max_new_tokens=100, max_bar_length=20):
        """
        Initialize the visualizer with a specified Hugging Face model.

        Args:
            model_name (str): Name of the Hugging Face model to load
            max_new_tokens (int): Maximum number of tokens to generate
            max_bar_length (int): Maximum length of activation bar in visualization
        """

        self.console = Console()

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.max_new_tokens = max_new_tokens
        self.max_bar_length = max_bar_length

    def generate_with_visualization(self, prompt):
        """
        Generate text and visualize model activations dynamically.

        Args:
            prompt (str): Initial text prompt to start generation
        """

        inputs = self.tokenizer(prompt, return_tensors="pt")
        generated_ids = inputs["input_ids"].tolist()[0]

        for _ in range(self.max_new_tokens):

            with torch.no_grad():
                outputs = self.model(torch.tensor([generated_ids]))
                logits = outputs.logits

                try:
                    hidden_states = outputs.hidden_states
                    attentions = outputs.attentions
                except AttributeError:
                    print(
                        "Model does not support hidden state or attention visualization."
                    )
                    break

            next_token_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
            top_probs, top_ids = torch.topk(next_token_probs, 10)
            next_token_id = top_ids[0].item()
            generated_ids.append(next_token_id)

            try:
                mlp_activations = self._process_mlp_activations(hidden_states)
                attn_activations = self._process_attention_activations(attentions)
            except Exception as e:
                print(f"Error processing activations: {e}")
                break

            self._render_visualization(
                generated_ids,
                next_token_id,
                mlp_activations,
                attn_activations,
                top_ids,
                top_probs,
                logits,
            )

            time.sleep(0.4)

    def _process_mlp_activations(self, hidden_states):
        """
        Process MLP (Feedforward) layer activations.

        Args:
            hidden_states (tuple): Hidden states from the model

        Returns:
            numpy.ndarray: Processed MLP activations
        """
        return np.array([layer[:, -1, :].mean().item() for layer in hidden_states])

    def _process_attention_activations(self, attentions):
        """
        Process self-attention layer activations.

        Args:
            attentions (tuple): Attention weights from the model

        Returns:
            numpy.ndarray: Processed attention activations
        """
        return np.array([attn[:, :, -1, :].mean().item() for attn in attentions])

    def _render_visualization(
        self,
        generated_ids,
        next_token_id,
        mlp_activations,
        attn_activations,
        top_ids,
        top_probs,
        logits,
    ):
        """
        Render the activation visualization using Rich library.

        Args:
            generated_ids (list): Generated token IDs
            next_token_id (int): Next predicted token ID
            mlp_activations (numpy.ndarray): MLP layer activations
            attn_activations (numpy.ndarray): Attention layer activations
            top_ids (torch.Tensor): Top predicted token IDs
            top_probs (torch.Tensor): Probabilities of top tokens
            logits (torch.Tensor): Raw logits
        """

        max_abs_act = max(
            np.max(np.abs(mlp_activations)), np.max(np.abs(attn_activations))
        )
        mlp_normalized = (mlp_activations / max_abs_act) * self.max_bar_length
        attn_normalized = (attn_activations / max_abs_act) * self.max_bar_length

        generated_text = self.tokenizer.decode(
            generated_ids[:-1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predicted_char = self.tokenizer.decode(
            [next_token_id], clean_up_tokenization_spaces=True
        )

        layout = Layout()

        activations_str = "[bold cyan]MAV [/bold cyan]\n\n"
        for i, (mlp_act, attn_act, raw_mlp, raw_attn) in enumerate(
            zip(mlp_normalized, attn_normalized, mlp_activations, attn_activations)
        ):
            mlp_bar = "█" * int(abs(mlp_act))
            mlp_color = "yellow" if raw_mlp >= 0 else "magenta"

            attn_bar = "█" * int(abs(attn_act))
            attn_color = "yellow" if raw_attn >= 0 else "magenta"

            activations_str += (
                f"[bold white]Layer {i:2d}[/] | "
                f"[bold yellow]MLP:[/] [{mlp_color}]{mlp_bar.ljust(self.max_bar_length)}[/] [bold yellow]{raw_mlp:+.4f}[/]  "
                f"[bold blue]ATTN:[/] [{attn_color}]{attn_bar.ljust(self.max_bar_length)}[/] [bold blue]{raw_attn:+.4f}[/]\n"
            )

        left_panel = Panel(
            activations_str, title="MLP & Attention Activations", border_style="cyan"
        )

        top_preds_str = "    ".join(
            f"[bold magenta]{self.tokenizer.decode([token_id], clean_up_tokenization_spaces=True)}[/] "
            f"([bold yellow]{prob:.1%}[/bold yellow], [bold cyan]{logit:.2f}[/bold cyan])"
            for token_id, prob, logit in zip(
                top_ids.tolist(), top_probs.tolist(), logits[0, -1, top_ids].tolist()
            )
        )

        right_panel = Panel(
            f"[bold blue]Top Predictions:[/bold blue]\n\n{top_preds_str}",
            title="Predictions",
            border_style="blue",
        )

        highlighted_text = Text(generated_text, style="bold bright_red")
        highlighted_text.append(predicted_char, style="bold on green")

        top_panel = Panel(
            highlighted_text, title="Generated Text", border_style="green"
        )

        layout.split_column(
            Layout(top_panel, size=12),
            Layout(right_panel, size=8),
            Layout(left_panel, size=20),
        )

        self.console.clear()
        self.console.print(layout)


def main():

    parser = argparse.ArgumentParser(description="Model Activation Visualizer")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model name (default: gpt2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a timeline ",
        help="Initial prompt for text generation",
    )
    parser.add_argument(
        "--tokens", type=int, default=100, help="Number of tokens to generate"
    )

    args = parser.parse_args()

    visualizer = ModelActivationVisualizer(
        model_name=args.model, max_new_tokens=args.tokens
    )
    visualizer.generate_with_visualization(args.prompt)


if __name__ == "__main__":
    main()
