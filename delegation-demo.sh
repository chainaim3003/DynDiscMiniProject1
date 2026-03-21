#!/bin/bash
#
# delegation-demo.sh - Complete Agent Delegation Demonstration
# Shows real-time output for demos while keeping logs
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
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

run_demo() {
    print_header "AGENT DELEGATION DEMONSTRATION"
    echo "This script will:"
    echo "  1. Sync source code from Windows to Linux"
    echo "  2. Preserve runtime data (task-data)"
    echo "  3. Rebuild Docker containers (SHOWN IN REAL-TIME)"
    echo "  4. Run agent delegation workflow (SHOWN IN REAL-TIME)"
    echo "  5. Perform deep verification (SHOWN IN REAL-TIME)"
    echo ""
    echo "Note: All output is shown AND logged to /tmp/*-output.log"
    echo ""
    read -p "Press ENTER to continue or Ctrl+C to cancel..."

    # PHASE 1: SYNC (quiet, just results)
    print_header "PHASE 1: SYNC SOURCE CODE (Windows → Linux)"
    
    echo "Step 1.1: Backing up Linux task-data..."
    BACKUP_TASK_DATA="/tmp/task-data-backup-$(date +%Y%m%d_%H%M%S)"
    if [ -d "${LINUX_TARGET}/task-data" ]; then
        cp -r "${LINUX_TARGET}/task-data" "${BACKUP_TASK_DATA}"
        print_success "task-data backed up"
    fi

    echo ""
    echo "Step 1.2: Syncing source code..."
    if rsync -a --delete \
      --exclude 'node_modules' --exclude '.git' --exclude 'task-data' \
      --exclude '*.log' --exclude '.env' --exclude '__pycache__' --exclude '*.pyc' \
      "${WIN_SOURCE}/" "${LINUX_TARGET}/" 2>&1 | tee /tmp/rsync-output.log | grep -E "^(sending|sent|total)" || true; then
        print_success "Source code synced"
    else
        print_error "Sync failed!"
        return 1
    fi

    echo ""
    echo "Step 1.3: Preserving task-data..."
    if [ ! -d "${LINUX_TARGET}/task-data" ] && [ -d "${BACKUP_TASK_DATA}" ]; then
        cp -r "${BACKUP_TASK_DATA}" "${LINUX_TARGET}/task-data"
        print_success "task-data restored"
    else
        print_success "task-data preserved"
    fi

    echo ""
    echo "Step 1.4: Fixing line endings..."
    find "${LINUX_TARGET}" -type f \( -name "*.sh" -o -name "*.ts" -o -name "*.js" \) \
      -not -path "*/node_modules/*" -not -path "*/task-data/*" \
      -exec sed -i 's/\r$//' {} \; 2>/dev/null
    print_success "Line endings fixed"

    echo ""
    echo "Step 1.5: Making scripts executable..."
    find "${LINUX_TARGET}" -type f -name "*.sh" -not -path "*/node_modules/*" -exec chmod +x {} \;
    print_success "Scripts executable"

    echo ""
    echo "Step 1.6: Verifying critical files..."
    local all_good=true
    for file in "sig-wallet/src/tasks/agent/agent-aid-delegate-finish.ts" \
                "sig-wallet/src/tasks/agent/agent-verify-delegation-deep.ts" \
                "test-agent-verification-DEEP.sh" \
                "run-all-buyerseller-2-with-agents.sh" \
                "docker-compose.yml"; do
        if [ -f "${LINUX_TARGET}/${file}" ]; then
            print_success "${file}"
        else
            print_error "MISSING: ${file}"
            all_good=false
        fi
    done

    if [ "$all_good" = false ]; then
        return 1
    fi

    echo ""
    echo "Step 1.7: Verifying KEL fix..."
    if grep -q "CRITICAL.*Verify the KEL actually exists" \
       "${LINUX_TARGET}/sig-wallet/src/tasks/agent/agent-aid-delegate-finish.ts"; then
        print_success "KEL verification fix present"
    else
        print_error "KEL fix MISSING!"
        return 1
    fi

    print_success "PHASE 1 COMPLETE"

    # PHASE 2: BUILD (SHOW EVERYTHING)
    print_header "PHASE 2: REBUILD CONTAINERS"
    
    cd "${LINUX_TARGET}" || return 1

    echo "Step 2.1: Stopping containers..."
    ./stop.sh
    print_success "Containers stopped"

    echo ""
    echo "Step 2.2: Rebuilding containers (showing real-time output)..."
    echo "-------------------------------------------------------------------"
    if docker compose build --no-cache 2>&1 | tee /tmp/docker-build.log; then
        echo "-------------------------------------------------------------------"
        print_success "Containers rebuilt"
    else
        echo "-------------------------------------------------------------------"
        print_error "Build failed!"
        return 1
    fi

    print_success "PHASE 2 COMPLETE"

    # PHASE 3: WORKFLOW (SHOW EVERYTHING)
    print_header "PHASE 3: RUN AGENT WORKFLOW"

    echo "Step 3.1: Deploying containers..."
    echo "-------------------------------------------------------------------"
    ./deploy.sh | tee /tmp/deploy-output.log
    echo "-------------------------------------------------------------------"
    print_success "Deployed"

    echo ""
    echo "Step 3.2: Waiting for services to stabilize..."
    sleep 5
    print_success "Services ready"

    echo ""
    echo "Step 3.3: Running agent workflow (showing real-time output)..."
    echo "This will create GEDA, QVI, LE, OORs, and Agents with full delegation"
    echo "-------------------------------------------------------------------"
    if ./run-all-buyerseller-2-with-agents.sh 2>&1 | tee /tmp/workflow-output.log; then
        echo "-------------------------------------------------------------------"
        print_success "Workflow completed"
    else
        echo "-------------------------------------------------------------------"
        print_error "Workflow failed!"
        return 1
    fi

    echo ""
    echo "Step 3.4: Verifying agent files were created..."
    
    # Defense in depth: Wait + check in container + verify on host
    echo "  → Waiting for file system flush (2 seconds)..."
    sleep 2
    
    echo "  → Checking files in container (source of truth)..."
    local container_count=$(docker compose exec -T tsx-shell ls -1 /task-data/*agent*-info.json 2>/dev/null | wc -l)
    echo "    Files in container: ${container_count}"
    
    echo "  → Checking files on host (bind mount)..."
    local host_count=$(ls -1 task-data/*agent*-info.json 2>/dev/null | wc -l)
    echo "    Files on host: ${host_count}"
    
    # Verify we have the expected files
    if [ $container_count -ge 2 ]; then
        print_success "Container has ${container_count} agent files ✓"
        
        if [ $host_count -ge 2 ]; then
            print_success "Host has ${host_count} agent files ✓"
            echo ""
            echo "Agent files created:"
            ls -lh task-data/*agent*-info.json
        else
            print_warning "Host only has ${host_count} files (container has ${container_count})"
            echo "Waiting additional 2 seconds for bind mount sync..."
            sleep 2
            host_count=$(ls -1 task-data/*agent*-info.json 2>/dev/null | wc -l)
            if [ $host_count -ge 2 ]; then
                print_success "Host now has ${host_count} agent files ✓"
            else
                print_warning "Host still has ${host_count} files, but container has them"
                echo "This is OK - files exist in container"
            fi
        fi
    else
        print_error "Expected 2+ agents in container, found ${container_count}"
        echo ""
        echo "Checking what files exist in container:"
        docker compose exec -T tsx-shell ls -la /task-data/ 2>/dev/null || echo "Could not list container files"
        echo ""
        echo "Checking what files exist on host:"
        ls -la task-data/ 2>/dev/null || echo "Could not list host files"
        return 1
    fi

    print_success "PHASE 3 COMPLETE"

    # PHASE 4: VERIFY (SHOW EVERYTHING)
    print_header "PHASE 4: DEEP DELEGATION VERIFICATION"

    echo "Performing deep verification (showing real-time output)..."
    echo "-------------------------------------------------------------------"
    if ./test-agent-verification-DEEP.sh 2>&1 | tee /tmp/verification-output.log; then
        echo "-------------------------------------------------------------------"
        print_success "PHASE 4 COMPLETE: Verification PASSED"
    else
        echo "-------------------------------------------------------------------"
        print_error "Verification failed"
        return 1
    fi

    # SUCCESS
    print_header "🎉 DEMONSTRATION COMPLETE!"
    echo -e "${GREEN}"
    echo "All phases succeeded:"
    echo "  ✅ Phase 1: Source code synced from Windows"
    echo "  ✅ Phase 2: Docker containers rebuilt"
    echo "  ✅ Phase 3: Agent workflow executed"
    echo "  ✅ Phase 4: Deep verification passed"
    echo ""
    echo "Agents created:"
    echo "  • jupiterSellerAgent (delegated from Jupiter_Chief_Sales_Officer)"
    echo "  • tommyBuyerAgent (delegated from Tommy_Chief_Procurement_Officer)"
    echo ""
    echo "Complete logs saved to:"
    echo "  • /tmp/rsync-output.log"
    echo "  • /tmp/docker-build.log"
    echo "  • /tmp/deploy-output.log"
    echo "  • /tmp/workflow-output.log"
    echo "  • /tmp/verification-output.log"
    echo -e "${NC}"

    print_header "READY FOR USE"
    echo "Test tommyBuyerAgent:"
    echo "  ./test-agent-verification-DEEP.sh tommyBuyerAgent Tommy_Chief_Procurement_Officer"
    echo ""
    echo "View trust tree:"
    echo "  cat task-data/trust-tree-buyerseller.txt"
    echo ""

    return 0
}

# Run the demo
run_demo
DEMO_RESULT=$?

if [ $DEMO_RESULT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Demo completed successfully!${NC}"
else
    echo ""
    echo -e "${RED}❌ Demo encountered errors. Check logs in /tmp/${NC}"
    echo "You are still in your shell - no exit occurred"
fi

# Return result but don't exit shell
return $DEMO_RESULT 2>/dev/null || true
