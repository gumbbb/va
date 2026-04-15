.PHONY: blue llm warmup docker-up run

BLUE_REPO := /home/acevn/ai-demo-edge-main

# 1) Connect Bluetooth speaker using local setup_blue.sh
blue:
	chmod +x ./setup_blue.sh
	./setup_blue.sh

# 2) Start LLM server using existing demo repo (assumes its Makefile/run.sh handle details)
llm:
	$(MAKE) -C $(BLUE_REPO) run

# 3) Send a few warmup requests to the LLM HTTP API (llama-server on :8080)
warmup:
	sleep 10
	@echo "Warming up LLM (1/3)..."
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{ "model": "llama", "messages": [ {"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"Say hello briefly."} ] }' >/dev/null || true
	@echo "Warming up LLM (2/3)..."
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{ "model": "llama", "messages": [ {"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"What is 2 plus 2?"} ] }' >/dev/null || true
	@echo "Warming up LLM (3/3)..."
	@curl -s -X POST http://localhost:8080/v1/chat/completions \
	  -H "Content-Type: application/json" \
	  -d '{ "model": "llama", "messages": [ {"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":"Give me a very short fun fact."} ] }' >/dev/null || true

# 4) Bring up full edge stack
docker-up:
	docker compose -p orch_v1 up

# Full sequence: BT -> LLM -> warmup -> wait -> docker compose up
run: blue
	( $(MAKE) -C $(BLUE_REPO) run ) &
	$(MAKE) warmup
	sleep 30
	docker compose -p orch_v1 up

