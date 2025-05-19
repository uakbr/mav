# MAV Web - Model Activity Visualizer UI

Web interface for visualizing internal model state during LLM text generation.

## Getting Started

### Prerequisites

- Node.js (v18+)
- npm or yarn
- A Vercel account (for deployment)
- Python 3.10 (for local testing)

### Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Deployment to Vercel

### One-Click Deploy

The easiest way to deploy this app is with Vercel's one-click deploy:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/yourusername/mav-web)

### Manual Deployment

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Log in to Vercel:
   ```bash
   vercel login
   ```

3. Deploy the app:
   ```bash
   vercel --prod
   ```

### Environment Variables

The following environment variables can be set in Vercel:

- `MAX_DURATION`: Maximum duration for serverless functions (recommended: 60)

## Architecture

### Frontend

- Next.js React application
- Tailwind CSS for styling
- Real-time visualization components

### Backend

- Python Serverless Functions on Vercel
- Event stream API for real-time updates
- Model caching for faster cold starts

## Resource Constraints

Be aware of the following resource constraints when deploying to Vercel:

- **Memory Limit**: Serverless functions have a 1024MB memory limit, so large models may not fit
- **Deployment Size**: The deployment size is limited to 50MB, so we use minimal dependencies
- **Timeout**: Free tier has a 10-second timeout for functions, Pro plan extends to 60 seconds

## Large Model Support

For larger models that exceed Vercel's limits, consider:

1. Using an external GPU service like Replicate
2. Setting up a dedicated GPU server and connecting via API
3. Pre-computing a set of visualizations and storing them statically

## License

This project is licensed under the MIT License - see the LICENSE file for details.
