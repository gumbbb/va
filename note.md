jetson-containers run \
  -e HUGGINGFACE_TOKEN=hf_xxx \
  -e FORCE_BUILD=on \
  -v $(pwd)/jetson_test/llama_gptq_free.sh:/opt/llama_gptq_free.sh \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /opt/llama_gptq_free.sh

# Benchmark TTFT & Tokens/sec:
jetson-containers run \
  -v $(pwd)/jetson_test/benchmark_ttft_tps.sh:/opt/benchmark_ttft_tps.sh \
  dustynv/tensorrt_llm:0.12-r36.4.0 \
  bash /opt/benchmark_ttft_tps.sh

  cd /home/acevn/ai-demo-edge-main/llama-cpp/models

wget "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf" \
  -O qwen3-0.6b-q4_k_m.gguf

hf download   bartowski/Llama-3.2-1B-Instruct-GGUF   --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf"   --local-dir .

source venv/bin/activate

cd /home/acevn/ai-demo-edge-main/jetson-containers

./jetson-containers run \
  -v /home/acevn/ai-demo-edge-main/llama-cpp/models:/models \
  $(./autotag llama_cpp)

llama-server \
  -m /models/llama-3.2-3b-instruct-q4_k_m.gguf \
  --jinja \
  --port 8080 \
  --host 0.0.0.0 \
  -ngl 99 \
  -t 12 \
  -fa \
  --log-colors \
  --parallel 1 \
  --n-predict 4096 \
  --ctx-size 8192 \
  --batch-size 8192 \
  --cpu-range 0-11 \
  --cpu-strict 1

  docker compose up orchestrator

   docker compose -p orch_v1 up car_manual

   docker compose -p orch_v1 build --no-cache
   
   car_manual-1  | 02:54:31 | INFO     | sentence_transformers.SentenceTransformer | Load pretrained SentenceTransformer: paraphrase-multilingual-MiniLM-L12-v2
car_manual-1  | '[Errno -3] Temporary failure in name resolution' thrown while requesting HEAD https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/adapter_config.json
car_manual-1  | 02:55:35 | WARNING  | huggingface_hub.utils._http | '[Errno -3] Temporary failure in name resolution' thrown while requesting HEAD https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/adapter_config.json
car_manual-1  | Retrying in 1s [Retry 1/5].
car_manual-1  | 02:55:35 | WARNING  | huggingface_hub.utils._http | Retrying in 1s [Retry 1/5].
car_manual-1  | Traceback (most recent call last):
car_manual-1  |   File "/app/main.py", line 20, in <module>
car_manual-1  |     executor.warm_up()
car_manual-1  |     ~~~~~~~~~~~~~~~~^^
car_manual-1  |   File "/app/agent_executor.py", line 87, in warm_up
car_manual-1  |     self._ensure_store()
car_manual-1  |     ~~~~~~~~~~~~~~~~~~^^
car_manual-1  |   File "/app/agent_executor.py", line 82, in _ensure_store
car_manual-1  |     self._store.load()
car_manual-1  |     ~~~~~~~~~~~~~~~~^^
car_manual-1  |   File "/app/qa_store.py", line 285, in load
car_manual-1  |     self._emb.eager_load()
car_manual-1  |     ~~~~~~~~~~~~~~~~~~~~^^
car_manual-1  |   File "/app/qa_store.py", line 144, in eager_load
car_manual-1  |     self._get_model()
car_manual-1  |     ~~~~~~~~~~~~~~~^^
car_manual-1  |   File "/app/qa_store.py", line 136, in _get_model
car_manual-1  |     self._model = SentenceTransformer(self._model_name, cache_folder=str(CACHE_DIR))
car_manual-1  |                   ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/sentence_transformers/SentenceTransformer.py", line 327, in __init__
car_manual-1  |     modules, self.module_kwargs = self._load_sbert_model(
car_manual-1  |                                   ~~~~~~~~~~~~~~~~~~~~~~^
car_manual-1  |         model_name_or_path,
car_manual-1  |         ^^^^^^^^^^^^^^^^^^^
car_manual-1  |     ...<7 lines>...
car_manual-1  |         config_kwargs=config_kwargs,
car_manual-1  |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |     )
car_manual-1  |     ^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/sentence_transformers/SentenceTransformer.py", line 2305, in _load_sbert_model
car_manual-1  |     module = module_class.load(
car_manual-1  |         model_name_or_path,
car_manual-1  |     ...<11 lines>...
car_manual-1  |         backend=self.backend,
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/sentence_transformers/models/Transformer.py", line 366, in load
car_manual-1  |     return cls(model_name_or_path=model_name_or_path, **init_kwargs)
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/sentence_transformers/models/Transformer.py", line 88, in __init__
car_manual-1  |     config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)
car_manual-1  |                             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/sentence_transformers/models/Transformer.py", line 139, in _load_config
car_manual-1  |     find_adapter_config_file(
car_manual-1  |     ~~~~~~~~~~~~~~~~~~~~~~~~^
car_manual-1  |         model_name_or_path,
car_manual-1  |         ^^^^^^^^^^^^^^^^^^^
car_manual-1  |     ...<3 lines>...
car_manual-1  |         local_files_only=config_args.get("local_files_only", False),
car_manual-1  |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |     )
car_manual-1  |     ^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/transformers/utils/peft_utils.py", line 83, in find_adapter_config_file
car_manual-1  |     adapter_cached_filename = cached_file(
car_manual-1  |         model_id,
car_manual-1  |     ...<11 lines>...
car_manual-1  |         _raise_exceptions_for_connection_errors=False,
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/transformers/utils/hub.py", line 276, in cached_file
car_manual-1  |     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/transformers/utils/hub.py", line 507, in cached_files
car_manual-1  |     raise e
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/transformers/utils/hub.py", line 419, in cached_files
car_manual-1  |     hf_hub_download(
car_manual-1  |     ~~~~~~~~~~~~~~~^
car_manual-1  |         path_or_repo_id,
car_manual-1  |         ^^^^^^^^^^^^^^^^
car_manual-1  |     ...<9 lines>...
car_manual-1  |         local_files_only=local_files_only,
car_manual-1  |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |     )
car_manual-1  |     ^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
car_manual-1  |     return fn(*args, **kwargs)
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/file_download.py", line 982, in hf_hub_download
car_manual-1  |     return _hf_hub_download_to_cache_dir(
car_manual-1  |         # Destination
car_manual-1  |     ...<15 lines>...
car_manual-1  |         dry_run=dry_run,
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/file_download.py", line 1115, in _hf_hub_download_to_cache_dir
car_manual-1  |     _get_metadata_or_catch_error(
car_manual-1  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
car_manual-1  |         repo_id=repo_id,
car_manual-1  |         ^^^^^^^^^^^^^^^^
car_manual-1  |     ...<10 lines>...
car_manual-1  |         retry_on_errors=True,
car_manual-1  |         ^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |     )
car_manual-1  |     ^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/file_download.py", line 1649, in _get_metadata_or_catch_error
car_manual-1  |     metadata = get_hf_file_metadata(
car_manual-1  |         url=url,
car_manual-1  |     ...<4 lines>...
car_manual-1  |         retry_on_errors=retry_on_errors,
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/utils/_validators.py", line 89, in _inner_fn
car_manual-1  |     return fn(*args, **kwargs)
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/file_download.py", line 1572, in get_hf_file_metadata
car_manual-1  |     response = _httpx_follow_relative_redirects_with_backoff(
car_manual-1  |         method="HEAD", url=url, headers=hf_headers, timeout=timeout, retry_on_errors=retry_on_errors
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/utils/_http.py", line 644, in _httpx_follow_relative_redirects_with_backoff
car_manual-1  |     response = http_backoff(
car_manual-1  |         method=method,
car_manual-1  |     ...<3 lines>...
car_manual-1  |         **no_retry_kwargs,
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/utils/_http.py", line 518, in http_backoff
car_manual-1  |     return next(
car_manual-1  |         _http_backoff_base(
car_manual-1  |     ...<9 lines>...
car_manual-1  |         )
car_manual-1  |     )
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/huggingface_hub/utils/_http.py", line 426, in _http_backoff_base
car_manual-1  |     response = client.request(method=method, url=url, **kwargs)
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/httpx/_client.py", line 825, in request
car_manual-1  |     return self.send(request, auth=auth, follow_redirects=follow_redirects)
car_manual-1  |            ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
car_manual-1  |   File "/app/.venv/lib/python3.14/site-packages/httpx/_client.py", line 901, in send
car_manual-1  |     raise RuntimeError("Cannot send a request, as the client has been closed.")
car_manual-1  | RuntimeError: Cannot send a request, as the client has been closed.
car_manual-1 exited with code 1

   
  
