# 🧩 vLEI Hackathon 2025 — Local Environment Setup Guide

This document provides a **complete step-by-step guide** to set up, clean, and deploy the **vLEI Hackathon 2025 Workshop Environment** using **Docker** on **Ubuntu / WSL2**.

---

## ⚙️ Prerequisites

Ensure the following are installed before proceeding:
- **Docker** and **Docker Compose**
- **Git**
- **WSL2 (Ubuntu)** if using Windows

---

## 🏗️ Step 1: Create Project Workspace and Copy Files

```bash
# Create a 'projects' folder in your home directory
mkdir -p ~/projects

# Copy the vLEI Hackathon project from your Windows folder into your Linux workspace
cp -r /mnt/c/CHAINAIM3003/mcp-servers/vLEINew1/vlei-hackathon-2025-workshop-master ~/projects/

(subject to change according to your windows file system location of the directory)

# Navigate to the working directory
cd ~/projects/vLEIWorkLinux1


🧰 Step 2: Install Required Tools
sudo apt update
sudo apt install dos2unix

🧹 Step 3: Fix Line Endings in Shell Scripts
find . -type f -name "*.sh" -exec dos2unix {} \;

🔑 Step 4: Make All Shell Scripts Executable
chmod +x *.sh
chmod +x scripts/*.sh 2>/dev/null
chmod +x */*.sh 2>/dev/null

📦 Step 5: Install jq JSON Processor
sudo apt update
sudo apt install jq -y

🧽 Step 6: Stop, Clean, Rebuild, and Deploy
# Stop any existing containers or services
./stop.sh

# Remove unused Docker containers, images, and networks
docker system prune -f

# Rebuild Docker images without cache for a fresh setup
docker compose build --no-cache

# Deploy the environment
./deploy.sh

🚀 Step 7: Run the Full vLEI Workflow
./run-all.sh
