# Deploying MAV Web to Vercel

This document provides detailed steps for deploying the MAV Web application to Vercel.

## Prerequisites

- GitHub account (for repository storage)
- Vercel account (for deployment)
- Node.js and npm installed locally (for development)

## Deployment Steps

### 1. Push to GitHub

1. Create a new GitHub repository
2. Initialize and push the project:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/mav-web.git
   git push -u origin main
   ```

### 2. Deploy to Vercel

#### Option A: Using Vercel Dashboard

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - Framework Preset: Next.js
   - Root Directory: ./
   - Build Command: npm run build
   - Install Command: npm install
   - Output Directory: .next
5. Add environment variables if needed
6. Click "Deploy"

#### Option B: Using Vercel CLI

1. Install Vercel CLI (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. Log in to Vercel:
   ```bash
   vercel login
   ```

3. Deploy from the project directory:
   ```bash
   cd mav-web
   vercel --prod
   ```

### 3. Configure Build Settings

When deploying, Vercel will automatically detect the Next.js project and set appropriate build settings. However, you may need to customize some settings:

1. In the Vercel dashboard, navigate to your project
2. Go to "Settings" → "General"
3. Scroll to "Build & Development Settings"
4. Ensure Next.js is selected as the framework

### 4. Configure Python Serverless Function

Vercel will automatically detect the Python serverless function in `vercel_python_api/run.py` and deploy it with the Python 3.10 runtime.

If needed, increase the function timeout in "Settings" → "Functions" → "Function Execution Timeout".

### 5. Monitoring the Deployment

1. After deployment, monitor the function logs:
   - Go to "Deployments" in the Vercel dashboard
   - Select the latest deployment
   - Click "Functions" tab
   - Look for the `/api/run` function

2. Check for errors in the logs if the visualization isn't working properly.

### 6. Resource Considerations

Be aware of Vercel's resource limitations:

1. **Memory**: Serverless functions are limited to 1024MB of memory
   - For large models, consider external GPU providers

2. **Execution Time**: 
   - Hobby tier: 10 seconds
   - Pro tier: 60 seconds
   - Enterprise tier: 900 seconds

3. **Deployment Size**: Limited to 50MB 
   - This should be sufficient for the model-less deployment (using external model hosts)

### 7. Custom Domain Setup (Optional)

1. Go to "Settings" → "Domains"
2. Add your custom domain
3. Follow the DNS configuration instructions

## Troubleshooting

If you encounter issues with the deployment:

1. **Function timeouts**: Increase the function execution timeout in settings
2. **Memory issues**: Reduce model size or switch to an external model provider
3. **Cold start performance**: Upgrade to a higher tier or implement model caching

## Maintenance

To update the deployed application:

1. Push changes to your GitHub repository
2. Vercel will automatically deploy the updates (if auto-deployment is enabled)
3. Or manually trigger a new deployment from the Vercel dashboard 