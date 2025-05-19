import json
import os
import time
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from openmav.backends.model_backend_transformers import TransformersBackend
from openmav.processors.state_fetcher import StateFetcher
import numpy as np
import torch

# Global cache for backends to reuse across invocations
backend_cache = {}

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def serialize_measurement(measurement):
    """Convert a ModelMeasurement object to a serializable dict"""
    return {
        'mlp_activations': measurement.mlp_activations,
        'mlp_normalized': measurement.mlp_normalized,
        'attention_entropy_values': measurement.attention_entropy_values,
        'attention_entropy_values_normalized': measurement.attention_entropy_values_normalized,
        'generated_text': measurement.generated_text,
        'predicted_char': measurement.predicted_char,
        'next_token_probs': measurement.next_token_probs,
        'top_ids': measurement.top_ids,
        'top_probs': measurement.top_probs,
        'logits': measurement.logits,
        'decoded_tokens': measurement.decoded_tokens
    }

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse URL and query parameters
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        
        # Extract parameters
        try:
            params_json = query_params.get('params', ['{}'])[0]
            data = json.loads(params_json)
            self._handle_request(data)
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Invalid parameters: {str(e)}'}).encode())
    
    def do_POST(self):
        # Read request body
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            self._handle_request(data)
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': f'Invalid JSON: {str(e)}'}).encode())
        
    def _handle_request(self, data):
        # Extract parameters
        model = data.get('model', 'gpt2')
        prompt = data.get('prompt', 'Hello world')
        max_new_tokens = int(data.get('max_new_tokens', 50))
        temperature = float(data.get('temperature', 0.0))
        top_k = int(data.get('top_k', 50))
        top_p = float(data.get('top_p', 1.0))
        min_p = float(data.get('min_p', 0.0))
        repetition_penalty = float(data.get('repetition_penalty', 1.0))
        refresh_rate = float(data.get('refresh_rate', 0.1))
        aggregation = data.get('aggregation', 'l2')
        scale = data.get('scale', 'linear')
        max_bar_length = int(data.get('max_bar_length', 35))
        device = "cpu"  # Force CPU for Vercel

        # Setup SSE response
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Get or create state fetcher
        cache_key = f"{model}"
        if cache_key not in backend_cache:
            try:
                backend = TransformersBackend(model_name=model, device=device, seed=42)
                state_fetcher = StateFetcher(
                    backend,
                    max_new_tokens=max_new_tokens,
                    aggregation=aggregation,
                    scale=scale,
                    max_bar_length=max_bar_length
                )
                backend_cache[cache_key] = {
                    "backend": backend,
                    "state_fetcher": state_fetcher
                }
            except Exception as e:
                error_msg = f"Error initializing model: {str(e)}"
                self.wfile.write(f"data: {json.dumps({'error': error_msg})}\n\n".encode())
                return

        state_fetcher = backend_cache[cache_key]["state_fetcher"]

        # Generate tokens and stream results
        try:
            for measurement in state_fetcher.fetch_next(
                prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty
            ):
                # Convert to JSON and send
                json_data = json.dumps(
                    serialize_measurement(measurement),
                    cls=NumpyJSONEncoder
                )
                self.wfile.write(f"data: {json_data}\n\n".encode())
                self.wfile.flush()
                
                # Respect refresh rate for UI updates
                time.sleep(refresh_rate)
                
                # Check if we've reached max tokens
                if len(measurement.generated_text.split()) >= max_new_tokens:
                    break
                    
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.wfile.write(f"data: {json.dumps({'error': error_msg})}\n\n".encode())
        
        # Signal end of stream
        self.wfile.write(f"data: {json.dumps({'done': True})}\n\n".encode())

def handler(request):
    if request.method == 'GET':
        return Handler().do_GET()
    else:
        return Handler().do_POST() 