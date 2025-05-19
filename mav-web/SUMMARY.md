# MAV Web - Implementation Summary

## Overview

MAV Web is a web-based implementation of the Model Activity Visualizer (MAV) tool, deployed as a Vercel application with a Next.js frontend and Python serverless function backend. This application allows users to visualize the internal workings of Large Language Models (LLMs) during text generation directly in their browser.

## Architecture

### Frontend (Next.js)

- **React Components**: A set of components that replicate the functionality of the original MAV terminal-based UI
- **Real-time Updates**: Uses Server-Sent Events (SSE) to receive streaming updates from the model
- **Interactive UI**: Form controls for model settings and parameters
- **Responsive Layout**: Grid-based layout that adapts to different screen sizes

### Backend (Python Serverless Function)

- **Vercel Serverless Function**: A Python endpoint that initializes and runs the OpenMAV library
- **Model Caching**: Caches model instances to reduce cold start times on subsequent requests
- **Stream Processing**: Streams model data in real-time using text/event-stream format
- **Resource Efficiency**: Optimized to run within Vercel's memory and execution time constraints

## Key Features

1. **Live Model Visualization**: Visualize the model's internal state (MLP activations, attention entropy, etc.) as it generates text
2. **Configurable Settings**: Adjust temperature, top-k, top-p, and other generation parameters
3. **Multiple Model Support**: Choose from different models (GPT-2, SmolLM, etc.)
4. **Real-time Updates**: See changes in the model's behavior as it generates text, token by token
5. **Responsive Design**: Works on various screen sizes, from desktop to tablet

## Implementation Highlights

### Components

- **MAVContainer**: Main container component that organizes and renders the visualization panels
- **Panel Components**: 
  - GeneratedTextPanel: Shows the generated text with the latest token highlighted
  - TopPredictionsPanel: Displays the top predicted tokens and their probabilities
  - MlpActivationsPanel: Visualizes MLP activations across model layers
  - AttentionEntropyPanel: Shows entropy values for attention matrices
  - OutputDistributionPanel: Displays the probability distribution across tokens

### Python Backend

- Uses the OpenMAV library for model initialization and token generation
- Implements a streaming API using Server-Sent Events (SSE)
- Handles both GET and POST requests for flexibility
- Implements error handling and graceful degradation

## Deployment Architecture

The application follows a serverless architecture on Vercel:

1. **Edge Network**: Vercel CDN serves static assets (HTML, CSS, JS)
2. **Serverless API**: Python function executes model inference
3. **Event Stream**: Real-time data flows from the API to the browser
4. **Model Caching**: Optimizes for repeated requests from the same client

## Resource Considerations

The implementation carefully considers Vercel's resource limitations:

- **Memory Usage**: ~350MB for GPT-2 small model
- **Deployment Size**: <50MB with minimal dependencies
- **Execution Time**: Uses streaming to stay within execution time limits
- **Fallback Options**: Documentation includes options for external GPU hosting for larger models

## Future Enhancements

1. **External GPU Support**: Add support for offloading to external GPU providers for larger models
2. **User Authentication**: Add user accounts to save and share visualizations
3. **Recording/Playback**: Allow recording and playback of model visualizations
4. **Custom Panels**: Support for user-defined custom panels
5. **Multi-model Comparison**: Compare multiple models side-by-side 