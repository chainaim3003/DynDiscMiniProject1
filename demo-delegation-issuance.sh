#!/bin/bash
#
# demo-delegation-issuance.sh - Agent Delegation Issuance Only
# This script syncs code, rebuilds, deploys, and runs the workflow
# It does NOT verify files - use demo-delegation-deep-verification.sh for that
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

WIN_SOURCE="/mnt/c/SATHYA/CHAINAIM3003/mcp-servers/stellarboston/vLEI1/vLEIWorkLinux1"
LINUX_TARGET="$HOME/projects/vLEIWorkLinux1"

print_header() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

run_issuance() {
    print_header "AGENT DELEGATION DEMONSTRATION - PART 1: ISSUANCE"
    echo "This script will:"
    echo "  1. Sync source code from Windows to Linux"
    echo "  2. Preserve runtime data (task-data)"
    echo "  3. Rebuild Docker containers"
    echo "  4. Deploy services"
    echo "  5. Run agent delegation workflow"
    echo ""
    echo "After completion, run: ./demo-delegation-deep-verification.sh"
    echo ""
    read -p "Press ENTER to continue or Ctrl+C to cancel..."

    # PHASE 1: SYNC
    print_header "PHASE 1: SYNC SOURCE CODE"
    
    echo "Backing up task-data..."
    BACKUP_TASK_DATA="/tmp/task-data-backup-$(date +%Y%m%d_%H%M%S)"
    if [ -d "${LINUX_TARGET}/task-data" ]; then
        cp -r "${LINUX_TARGET}/task-data" "${BACKUP_TASK_DATA}"
    fi

    echo "Syncing from Windows..."
    rsync -a --delete \
      --exclude 'node_modules' --exclude '.git' --exclude 'task-data' \
      --exclude '*.log' --exclude '.env' --exclude '__pycache__' --exclude '*.pyc' \
      "${WIN_SOURCE}/" "${LINUX_TARGET}/" > /tmp/rsync-output.log 2>&1

    if [ ! -d "${LINUX_TARGET}/task-data" ] && [ -d "${BACKUP_TASK_DATA}" ]; then
        cp -r "${BACKUP_TASK_DATA}" "${LINUX_TARGET}/task-data"
    fi

    echo "Fixing line endings..."
    find "${LINUX_TARGET}" -type f \( -name "*.sh" -o -name "*.ts" -o -name "*.js" \) \
      -not -path "*/node_modules/*" -not -path "*/task-data/*" \
      -exec sed -i 's/\r$//' {} \; 2>/dev/null

    find "${LINUX_TARGET}" -type f -name "*.sh" -not -path "*/node_modules/*" -exec chmod +x {} \;

    print_success "PHASE 1 COMPLETE"

    # PHASE 2: BUILD
    print_header "PHASE 2: REBUILD CONTAINERS"
    
    cd "${LINUX_TARGET}" || return 1

    echo "Stopping containers..."
    ./stop.sh
    
    echo ""
    echo "Rebuilding (this takes time)..."
    echo "-------------------------------------------------------------------"
    if docker compose build --no-cache 2>&1 | tee /tmp/docker-build.log; then
        echo "-------------------------------------------------------------------"
        print_success "PHASE 2 COMPLETE"
    else
        print_error "Build failed!"
        return 1
    fi

    # PHASE 3: DEPLOY & WORKFLOW
    print_header "PHASE 3: DEPLOY AND RUN WORKFLOW"

    echo "Deploying containers..."
    echo "-------------------------------------------------------------------"
    ./deploy.sh | tee /tmp/deploy-output.log
    echo "-------------------------------------------------------------------"
    
    echo ""
    echo "Waiting for services..."
    sleep 5

    echo ""
    echo "Running agent delegation workflow..."
    echo "-------------------------------------------------------------------"
    if ./run-all-buyerseller-2-with-agents.sh 2>&1 | tee /tmp/workflow-output.log; then
        echo "-------------------------------------------------------------------"
        print_success "PHASE 3 COMPLETE"
    else
        print_error "Workflow failed!"
        return 1
    fi

    # SUCCESS
    print_header "🎉 ISSUANCE COMPLETE!"
    echo -e "${GREEN}"
    echo "All phases succeeded:"
    echo "  ✅ Source code synced"
    echo "  ✅ Containers rebuilt"
    echo "  ✅ Services deployed"
    echo "  ✅ Workflow executed"
    echo ""
    echo "Agents created:"
    echo "  • jupiterSellerAgent"
    echo "  • tommyBuyerAgent"
    echo ""
    echo "Logs saved to:"
    echo "  • /tmp/rsync-output.log"
    echo "  • /tmp/docker-build.log"
    echo "  • /tmp/deploy-output.log"
    echo "  • /tmp/workflow-output.log"
    echo -e "${NC}"

    print_header "NEXT STEP"
    echo "Run deep verification:"
    echo "  ${BLUE}./demo-delegation-deep-verification.sh${NC}"
    echo ""

    return 0
}

# Run issuance
run_issuance
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Issuance completed successfully!${NC}"
else
    echo -e "${RED}❌ Issuance encountered errors.${NC}"
fi

return $RESULT 2>/dev/null || true
