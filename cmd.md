./stop.sh
./setup.sh

<!-- docker system prune -f
docker compose build --no-cache -->
./deploy.sh
./saidify-and-restart.sh

<!-- ./run_4c.sh -->
./run-all-buyerseller-4D-with-subdelegation.sh

./run-all-buyerseller-4C-with-agents.sh
./DEEP-EXT-credential.sh

[./task-scripts/subagent/generate-unique-subagent-brans.sh]

./generate-subagent-brans.sh

./task-scripts/subagent/subagent-delegate-with-unique-bran.sh \
    JupiterTreasuryAgent \
    jupiterSellerAgent