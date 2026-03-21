# Dockerfile for gleif/tsx-shell image
# vLEI Workshop - TypeScript execution environment
#
# NOTE: The ./src directory is mounted as a volume in docker-compose.yml
# This allows live updates to TypeScript files without rebuilding.
# We create an empty src directory here as a placeholder.

FROM node:24-alpine3.20

# Install bash and curl for debugging
RUN apk add --no-cache \
    bash \
    curl

WORKDIR /vlei

# Copy package files and install dependencies
COPY package.json ./
COPY package-lock.json ./
COPY tsconfig.json ./
RUN npm install

# Create empty src directory (actual files come from volume mount)
# This ensures the directory exists even if volume mount fails
RUN mkdir -p ./src

# Install tsx globally for TypeScript execution
RUN npm install -g tsx

# Default command
CMD ["tsx"]
