#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MAV Web Deployment Script ===${NC}"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}Vercel CLI is not installed. Installing now...${NC}"
    npm install -g vercel
fi

# Check if user is logged in to Vercel
echo -e "${BLUE}Checking Vercel authentication...${NC}"
if ! vercel whoami &> /dev/null; then
    echo -e "${RED}Not logged in to Vercel. Please login:${NC}"
    vercel login
fi

# Build the application
echo -e "${BLUE}Building the application...${NC}"
npm run build

# Deploy to Vercel
echo -e "${BLUE}Deploying to Vercel...${NC}"
vercel --prod

echo -e "${GREEN}Deployment complete! Your MAV Web application should now be live.${NC}"
echo -e "${BLUE}Visit your Vercel dashboard to see the deployment details.${NC}" 