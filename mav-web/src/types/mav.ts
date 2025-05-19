export interface ModelMeasurement {
  mlp_activations: number[][];
  mlp_normalized: number[][];
  attention_entropy_values: number[][];
  attention_entropy_values_normalized: number[][];
  generated_text: string;
  predicted_char: string;
  next_token_probs: number[];
  top_ids: number[];
  top_probs: number[];
  logits: number[][][];
  decoded_tokens: string[];
}

export interface MAVParams {
  model: string;
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  repetition_penalty?: number;
  refresh_rate?: number;
  aggregation?: string;
  scale?: string;
  max_bar_length?: number;
}

export interface ErrorResponse {
  error: string;
}

export interface DoneResponse {
  done: boolean;
} 