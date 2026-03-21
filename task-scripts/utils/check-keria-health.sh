#!/bin/bash
################################################################################
# check-keria-health.sh
# Utility script to check KERIA health and wait until healthy
################################################################################

MAX_CHECKS="${1:-10}"
SLEEP_SECONDS="${2:-3}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "Checking KERIA health (max $MAX_CHECKS attempts)..."

for i in $(seq 1 $MAX_CHECKS); do
    # Check admin API (port 3901)
    if docker compose exec -T keria wget --spider --tries=1 --no-verbose --timeout=5 http://127.0.0.1:3901/health 2>/dev/null || \
       docker compose exec -T keria wget --spider --tries=1 --no-verbose --timeout=5 http://127.0.0.1:3902/spec.yaml 2>/dev/null; then
        echo -e "${GREEN}✓ KERIA is healthy (check $i)${NC}"
        exit 0
    else
        echo -e "${YELLOW}  KERIA health check $i/$MAX_CHECKS - waiting ${SLEEP_SECONDS}s...${NC}"
        
        # Also check if container is running
        if ! docker compose ps keria | grep -q "Up"; then
            echo -e "${RED}✗ KERIA container is not running!${NC}"
            echo "  Attempting to restart KERIA..."
            docker compose up -d keria
            sleep 5
        fi
        
        sleep $SLEEP_SECONDS
    fi
done

echo -e "${RED}✗ KERIA is not healthy after $MAX_CHECKS checks${NC}"
echo "  Showing KERIA logs:"
docker compose logs keria --tail=30
exit 1
